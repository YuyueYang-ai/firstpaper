/**
 * This file is part of OmniGS
 */

#include "include/geometry_codec.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace
{

std::uint32_t zigzagEncode(std::int32_t value)
{
    return static_cast<std::uint32_t>((value << 1) ^ (value >> 31));
}

std::int32_t zigzagDecode(std::uint32_t value)
{
    return static_cast<std::int32_t>((value >> 1) ^ (~(value & 1u) + 1u));
}

void writeVarUint(std::ofstream& out, std::uint32_t value)
{
    while (value >= 0x80u) {
        const std::uint8_t byte = static_cast<std::uint8_t>((value & 0x7Fu) | 0x80u);
        out.put(static_cast<char>(byte));
        value >>= 7u;
    }
    out.put(static_cast<char>(static_cast<std::uint8_t>(value)));
}

std::uint32_t readVarUint(std::ifstream& in)
{
    std::uint32_t value = 0u;
    int shift = 0;
    while (true) {
        const int byte = in.get();
        if (byte == EOF)
            throw std::runtime_error("geometry codec: truncated varint stream.");
        value |= static_cast<std::uint32_t>(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0)
            break;
        shift += 7;
        if (shift > 28)
            throw std::runtime_error("geometry codec: varint is too large.");
    }
    return value;
}

} // namespace

namespace geometry_codec
{

void encodeMortonDelta(
    const torch::Tensor& xyz,
    const std::filesystem::path& path,
    Json::Value& meta,
    const GeometryCodecOptions& options)
{
    auto xyz_cpu = xyz.detach().contiguous().to(torch::kCPU, torch::kFloat32);
    if (xyz_cpu.dim() != 2 || xyz_cpu.size(1) != 3)
        throw std::runtime_error("geometry codec expects xyz with shape [N, 3].");

    const auto num_points = xyz_cpu.size(0);
    const int quant_bits = std::clamp(options.quant_bits, 8, 24);
    const std::uint32_t qmax = (1u << quant_bits) - 1u;

    auto bbox_min = std::get<0>(xyz_cpu.min(0)).contiguous();
    auto bbox_max = std::get<0>(xyz_cpu.max(0)).contiguous();
    const float* bbox_min_ptr = bbox_min.data_ptr<float>();
    const float* bbox_max_ptr = bbox_max.data_ptr<float>();
    std::array<float, 3> min_v{bbox_min_ptr[0], bbox_min_ptr[1], bbox_min_ptr[2]};
    std::array<float, 3> max_v{bbox_max_ptr[0], bbox_max_ptr[1], bbox_max_ptr[2]};
    std::array<float, 3> range_v{
        std::max(max_v[0] - min_v[0], 1e-8f),
        std::max(max_v[1] - min_v[1], 1e-8f),
        std::max(max_v[2] - min_v[2], 1e-8f)};

    const float* xyz_ptr = xyz_cpu.data_ptr<float>();
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open geometry bitstream at " + path.string());

    std::array<std::uint32_t, 3> prev_q{0u, 0u, 0u};
    for (int64_t idx = 0; idx < num_points; ++idx) {
        std::array<std::uint32_t, 3> q{};
        for (int axis = 0; axis < 3; ++axis) {
            const float normalized = std::clamp((xyz_ptr[idx * 3 + axis] - min_v[axis]) / range_v[axis], 0.0f, 1.0f);
            q[axis] = static_cast<std::uint32_t>(std::llround(static_cast<double>(normalized) * static_cast<double>(qmax)));
        }
        if (idx == 0) {
            writeVarUint(out, q[0]);
            writeVarUint(out, q[1]);
            writeVarUint(out, q[2]);
        }
        else {
            writeVarUint(out, zigzagEncode(static_cast<std::int32_t>(q[0]) - static_cast<std::int32_t>(prev_q[0])));
            writeVarUint(out, zigzagEncode(static_cast<std::int32_t>(q[1]) - static_cast<std::int32_t>(prev_q[1])));
            writeVarUint(out, zigzagEncode(static_cast<std::int32_t>(q[2]) - static_cast<std::int32_t>(prev_q[2])));
        }
        prev_q = q;
    }

    meta["codec"] = "morton_delta_varint";
    meta["quant_bits"] = quant_bits;
    meta["num_points"] = Json::Value::Int64(num_points);
    Json::Value bbox_min_json(Json::arrayValue);
    Json::Value bbox_max_json(Json::arrayValue);
    for (int axis = 0; axis < 3; ++axis) {
        bbox_min_json.append(min_v[axis]);
        bbox_max_json.append(max_v[axis]);
    }
    meta["bbox_min"] = bbox_min_json;
    meta["bbox_max"] = bbox_max_json;
}

torch::Tensor decodeMortonDelta(
    const std::filesystem::path& path,
    const Json::Value& meta,
    torch::DeviceType device_type)
{
    const int quant_bits = meta["quant_bits"].asInt();
    const std::uint32_t qmax = (1u << quant_bits) - 1u;
    const auto num_points = meta["num_points"].asInt64();
    if (num_points < 0)
        throw std::runtime_error("geometry codec: invalid point count.");

    std::array<float, 3> min_v{};
    std::array<float, 3> max_v{};
    std::array<float, 3> range_v{};
    for (int axis = 0; axis < 3; ++axis) {
        min_v[axis] = meta["bbox_min"][axis].asFloat();
        max_v[axis] = meta["bbox_max"][axis].asFloat();
        range_v[axis] = std::max(max_v[axis] - min_v[axis], 1e-8f);
    }

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open geometry bitstream at " + path.string());

    auto xyz = torch::empty(
        {num_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto* xyz_ptr = xyz.data_ptr<float>();

    std::array<std::uint32_t, 3> q{};
    if (num_points > 0) {
        q[0] = readVarUint(in);
        q[1] = readVarUint(in);
        q[2] = readVarUint(in);
        for (int axis = 0; axis < 3; ++axis) {
            const float normalized = static_cast<float>(q[axis]) / static_cast<float>(qmax);
            xyz_ptr[axis] = min_v[axis] + normalized * range_v[axis];
        }
    }

    for (int64_t idx = 1; idx < num_points; ++idx) {
        q[0] = static_cast<std::uint32_t>(static_cast<std::int32_t>(q[0]) + zigzagDecode(readVarUint(in)));
        q[1] = static_cast<std::uint32_t>(static_cast<std::int32_t>(q[1]) + zigzagDecode(readVarUint(in)));
        q[2] = static_cast<std::uint32_t>(static_cast<std::int32_t>(q[2]) + zigzagDecode(readVarUint(in)));
        for (int axis = 0; axis < 3; ++axis) {
            const float normalized = static_cast<float>(q[axis]) / static_cast<float>(qmax);
            xyz_ptr[idx * 3 + axis] = min_v[axis] + normalized * range_v[axis];
        }
    }

    return xyz.to(device_type);
}

} // namespace geometry_codec
