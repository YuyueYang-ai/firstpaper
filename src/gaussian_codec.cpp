/**
 * This file is part of OmniGS
 *
 * Copyright (C) 2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * OmniGS is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * OmniGS is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with OmniGS.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * OmniGS is Derivative Works of Gaussian Splatting.
 * Its usage should not break the terms in LICENSE.md.
 */

#include "include/gaussian_codec.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <json/json.h>

namespace
{

template<typename T>
void writeBinary(const std::filesystem::path& path, const std::vector<T>& data)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open binary file for writing: " + path.string());
    if (!data.empty())
        out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
}

template<typename T>
std::vector<T> readBinary(const std::filesystem::path& path, std::size_t expected_count)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open binary file for reading: " + path.string());

    std::vector<T> data(expected_count);
    if (expected_count > 0)
        in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(expected_count * sizeof(T)));
    return data;
}

torch::Tensor toCpuFloat2D(const torch::Tensor& tensor, int64_t rows, int64_t cols)
{
    return tensor.detach().contiguous().view({rows, cols}).to(torch::kCPU, torch::kFloat32);
}

template<typename QType>
struct Quantized2D
{
    std::vector<QType> data;
    std::vector<float> mins;
    std::vector<float> maxs;
};

template<typename QType>
Quantized2D<QType> quantize2D(const torch::Tensor& tensor, int64_t rows, int64_t cols)
{
    auto tensor_cpu = toCpuFloat2D(tensor, rows, cols);
    const float* ptr = tensor_cpu.data_ptr<float>();
    Quantized2D<QType> quantized;
    quantized.data.resize(rows * cols);
    quantized.mins.assign(cols, std::numeric_limits<float>::max());
    quantized.maxs.assign(cols, std::numeric_limits<float>::lowest());

    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < cols; ++col) {
            const float value = ptr[row * cols + col];
            quantized.mins[col] = std::min(quantized.mins[col], value);
            quantized.maxs[col] = std::max(quantized.maxs[col], value);
        }
    }

    const double levels = static_cast<double>(std::numeric_limits<QType>::max());
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < cols; ++col) {
            const float min_value = quantized.mins[col];
            const float max_value = quantized.maxs[col];
            const float value = ptr[row * cols + col];
            QType q = 0;
            if (max_value - min_value > 1e-8f) {
                double normalized = static_cast<double>(value - min_value) / static_cast<double>(max_value - min_value);
                normalized = std::clamp(normalized, 0.0, 1.0);
                q = static_cast<QType>(std::llround(normalized * levels));
            }
            quantized.data[row * cols + col] = q;
        }
    }

    return quantized;
}

template<typename QType>
torch::Tensor dequantize2D(
    const std::vector<QType>& data,
    int64_t rows,
    int64_t cols,
    const std::vector<float>& mins,
    const std::vector<float>& maxs,
    torch::DeviceType device_type)
{
    std::vector<float> values(rows * cols, 0.0f);
    const double levels = static_cast<double>(std::numeric_limits<QType>::max());

    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < cols; ++col) {
            const float min_value = mins[col];
            const float max_value = maxs[col];
            float value = min_value;
            if (max_value - min_value > 1e-8f) {
                value = min_value + static_cast<float>(static_cast<double>(data[row * cols + col]) / levels) * (max_value - min_value);
            }
            values[row * cols + col] = value;
        }
    }

    auto tensor = torch::from_blob(
        values.data(),
        {rows, cols},
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
    return tensor.to(device_type);
}

template<typename QType>
struct Quantized1D
{
    std::vector<QType> data;
    float min_value = 0.0f;
    float max_value = 0.0f;
};

template<typename QType>
Quantized1D<QType> quantize1D(const std::vector<float>& values)
{
    Quantized1D<QType> quantized;
    quantized.data.resize(values.size(), 0);
    if (values.empty())
        return quantized;

    auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    quantized.min_value = *min_it;
    quantized.max_value = *max_it;

    const double levels = static_cast<double>(std::numeric_limits<QType>::max());
    if (quantized.max_value - quantized.min_value <= 1e-8f)
        return quantized;

    for (std::size_t idx = 0; idx < values.size(); ++idx) {
        double normalized = static_cast<double>(values[idx] - quantized.min_value)
                            / static_cast<double>(quantized.max_value - quantized.min_value);
        normalized = std::clamp(normalized, 0.0, 1.0);
        quantized.data[idx] = static_cast<QType>(std::llround(normalized * levels));
    }
    return quantized;
}

template<typename QType>
std::vector<float> dequantize1D(const std::vector<QType>& values, float min_value, float max_value)
{
    std::vector<float> decoded(values.size(), min_value);
    if (values.empty() || max_value - min_value <= 1e-8f)
        return decoded;

    const double levels = static_cast<double>(std::numeric_limits<QType>::max());
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
        decoded[idx] = min_value + static_cast<float>(static_cast<double>(values[idx]) / levels) * (max_value - min_value);
    }
    return decoded;
}

Json::Value vectorToJson(const std::vector<float>& values)
{
    Json::Value json(Json::arrayValue);
    for (float value : values)
        json.append(value);
    return json;
}

std::vector<float> jsonToVector(const Json::Value& json)
{
    std::vector<float> values;
    values.reserve(json.size());
    for (const auto& entry : json)
        values.push_back(entry.asFloat());
    return values;
}

std::vector<float> buildRestPayload(
    const torch::Tensor& features_rest,
    const torch::Tensor& sh_levels,
    int max_sh_degree)
{
    std::vector<float> payload;
    if (max_sh_degree <= 0 || !features_rest.defined() || features_rest.numel() == 0)
        return payload;

    auto rest_cpu = features_rest.detach().contiguous().to(torch::kCPU, torch::kFloat32);
    auto levels_cpu = sh_levels.detach().contiguous().to(torch::kCPU, torch::kInt32);
    const float* rest_ptr = rest_cpu.data_ptr<float>();
    const int32_t* levels_ptr = levels_cpu.data_ptr<int32_t>();
    const int64_t num_points = rest_cpu.size(0);
    const int64_t rest_coeffs = rest_cpu.size(1);

    std::size_t payload_size = 0;
    for (int64_t point_idx = 0; point_idx < num_points; ++point_idx) {
        const int level = std::clamp(levels_ptr[point_idx], 0, max_sh_degree);
        payload_size += static_cast<std::size_t>(((level + 1) * (level + 1) - 1) * 3);
    }
    payload.reserve(payload_size);

    for (int64_t point_idx = 0; point_idx < num_points; ++point_idx) {
        const int level = std::clamp(levels_ptr[point_idx], 0, max_sh_degree);
        const int coeff_count = (level + 1) * (level + 1) - 1;
        for (int coeff_idx = 0; coeff_idx < coeff_count; ++coeff_idx) {
            for (int channel = 0; channel < 3; ++channel) {
                payload.push_back(rest_ptr[(point_idx * rest_coeffs + coeff_idx) * 3 + channel]);
            }
        }
    }

    return payload;
}

torch::Tensor restoreRestPayload(
    const std::vector<float>& payload,
    const torch::Tensor& sh_levels,
    int64_t num_points,
    int max_sh_degree,
    torch::DeviceType device_type)
{
    const int64_t rest_coeffs = std::max(0, (max_sh_degree + 1) * (max_sh_degree + 1) - 1);
    auto rest = torch::zeros(
        {num_points, rest_coeffs, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type));
    if (rest_coeffs == 0 || payload.empty())
        return rest;

    auto rest_cpu = rest.to(torch::kCPU);
    auto levels_cpu = sh_levels.detach().contiguous().to(torch::kCPU, torch::kInt32);
    float* rest_ptr = rest_cpu.data_ptr<float>();
    const int32_t* levels_ptr = levels_cpu.data_ptr<int32_t>();

    std::size_t payload_offset = 0;
    for (int64_t point_idx = 0; point_idx < num_points; ++point_idx) {
        const int level = std::clamp(levels_ptr[point_idx], 0, max_sh_degree);
        const int coeff_count = (level + 1) * (level + 1) - 1;
        for (int coeff_idx = 0; coeff_idx < coeff_count; ++coeff_idx) {
            for (int channel = 0; channel < 3; ++channel) {
                rest_ptr[(point_idx * rest_coeffs + coeff_idx) * 3 + channel] = payload[payload_offset++];
            }
        }
    }

    return rest_cpu.to(device_type);
}

} // namespace

void GaussianCodec::save(
    const DecodedGaussianTensors& decoded,
    const std::filesystem::path& result_dir,
    const CompactExportOptions& options)
{
    if (decoded.empty())
        throw std::runtime_error("Cannot save an empty compact Gaussian package.");

    std::filesystem::create_directories(result_dir);

    const int64_t num_points = decoded.xyz.size(0);
    const auto sh_levels_u8 = decoded.sh_levels.to(torch::kCPU, torch::kUInt8).contiguous();
    std::vector<uint8_t> sh_levels(
        sh_levels_u8.data_ptr<uint8_t>(),
        sh_levels_u8.data_ptr<uint8_t>() + sh_levels_u8.numel());

    const bool use_attr_u16 = options.attribute_quant_bits > 8;
    const bool use_rot_u16 = options.rotation_quant_bits > 8;

    Json::Value meta;
    meta["num_points"] = static_cast<Json::UInt64>(num_points);
    meta["max_sh_degree"] = decoded.max_sh_degree;
    meta["active_sh_degree"] = decoded.active_sh_degree;
    meta["rest_payload_values"] = static_cast<Json::UInt64>(0);
    meta["rotation_storage"] = use_rot_u16 ? "uint16" : "uint8";
    meta["attribute_storage"] = use_attr_u16 ? "uint16" : "uint8";

    const auto xyz_quant = quantize2D<uint16_t>(decoded.xyz, num_points, 3);
    writeBinary(result_dir / "xyz.bin", xyz_quant.data);
    meta["xyz"]["mins"] = vectorToJson(xyz_quant.mins);
    meta["xyz"]["maxs"] = vectorToJson(xyz_quant.maxs);

    if (use_attr_u16) {
        const auto fdc_quant = quantize2D<uint16_t>(decoded.features_dc, num_points, 3);
        const auto opacity_quant = quantize2D<uint16_t>(decoded.opacity, num_points, 1);
        const auto scaling_quant = quantize2D<uint16_t>(decoded.scaling, num_points, 3);
        writeBinary(result_dir / "f_dc.bin", fdc_quant.data);
        writeBinary(result_dir / "opacity.bin", opacity_quant.data);
        writeBinary(result_dir / "scaling.bin", scaling_quant.data);
        meta["f_dc"]["mins"] = vectorToJson(fdc_quant.mins);
        meta["f_dc"]["maxs"] = vectorToJson(fdc_quant.maxs);
        meta["opacity"]["mins"] = vectorToJson(opacity_quant.mins);
        meta["opacity"]["maxs"] = vectorToJson(opacity_quant.maxs);
        meta["scaling"]["mins"] = vectorToJson(scaling_quant.mins);
        meta["scaling"]["maxs"] = vectorToJson(scaling_quant.maxs);
    }
    else {
        const auto fdc_quant = quantize2D<uint8_t>(decoded.features_dc, num_points, 3);
        const auto opacity_quant = quantize2D<uint8_t>(decoded.opacity, num_points, 1);
        const auto scaling_quant = quantize2D<uint8_t>(decoded.scaling, num_points, 3);
        writeBinary(result_dir / "f_dc.bin", fdc_quant.data);
        writeBinary(result_dir / "opacity.bin", opacity_quant.data);
        writeBinary(result_dir / "scaling.bin", scaling_quant.data);
        meta["f_dc"]["mins"] = vectorToJson(fdc_quant.mins);
        meta["f_dc"]["maxs"] = vectorToJson(fdc_quant.maxs);
        meta["opacity"]["mins"] = vectorToJson(opacity_quant.mins);
        meta["opacity"]["maxs"] = vectorToJson(opacity_quant.maxs);
        meta["scaling"]["mins"] = vectorToJson(scaling_quant.mins);
        meta["scaling"]["maxs"] = vectorToJson(scaling_quant.maxs);
    }

    if (use_rot_u16) {
        const auto rotation_quant = quantize2D<uint16_t>(decoded.rotation, num_points, 4);
        writeBinary(result_dir / "rotation.bin", rotation_quant.data);
        meta["rotation"]["mins"] = vectorToJson(rotation_quant.mins);
        meta["rotation"]["maxs"] = vectorToJson(rotation_quant.maxs);
    }
    else {
        const auto rotation_quant = quantize2D<uint8_t>(decoded.rotation, num_points, 4);
        writeBinary(result_dir / "rotation.bin", rotation_quant.data);
        meta["rotation"]["mins"] = vectorToJson(rotation_quant.mins);
        meta["rotation"]["maxs"] = vectorToJson(rotation_quant.maxs);
    }

    auto rest_payload = buildRestPayload(decoded.features_rest, decoded.sh_levels, decoded.max_sh_degree);
    meta["rest_payload_values"] = static_cast<Json::UInt64>(rest_payload.size());
    if (use_attr_u16) {
        const auto rest_quant = quantize1D<uint16_t>(rest_payload);
        writeBinary(result_dir / "f_rest.bin", rest_quant.data);
        meta["f_rest"]["min"] = rest_quant.min_value;
        meta["f_rest"]["max"] = rest_quant.max_value;
    }
    else {
        const auto rest_quant = quantize1D<uint8_t>(rest_payload);
        writeBinary(result_dir / "f_rest.bin", rest_quant.data);
        meta["f_rest"]["min"] = rest_quant.min_value;
        meta["f_rest"]["max"] = rest_quant.max_value;
    }

    writeBinary(result_dir / "sh_levels.bin", sh_levels);

    std::ofstream meta_out(result_dir / "metadata.json");
    if (!meta_out.is_open())
        throw std::runtime_error("Cannot open metadata file for writing: " + (result_dir / "metadata.json").string());

    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(meta, &meta_out);
}

DecodedGaussianTensors GaussianCodec::load(
    const std::filesystem::path& result_dir,
    torch::DeviceType device_type)
{
    std::ifstream meta_in(result_dir / "metadata.json");
    if (!meta_in.is_open())
        throw std::runtime_error("Cannot open compact metadata: " + (result_dir / "metadata.json").string());

    Json::Value meta;
    meta_in >> meta;

    const int64_t num_points = static_cast<int64_t>(meta["num_points"].asUInt64());
    const int max_sh_degree = meta["max_sh_degree"].asInt();
    const int active_sh_degree = meta["active_sh_degree"].asInt();
    const bool attr_u16 = meta["attribute_storage"].asString() == "uint16";
    const bool rot_u16 = meta["rotation_storage"].asString() == "uint16";

    auto sh_levels_raw = readBinary<uint8_t>(result_dir / "sh_levels.bin", static_cast<std::size_t>(num_points));
    auto sh_levels = torch::from_blob(
        sh_levels_raw.data(),
        {num_points},
        torch::TensorOptions().dtype(torch::kUInt8)).clone().to(torch::kInt32).to(device_type);

    auto xyz = dequantize2D<uint16_t>(
        readBinary<uint16_t>(result_dir / "xyz.bin", static_cast<std::size_t>(num_points * 3)),
        num_points,
        3,
        jsonToVector(meta["xyz"]["mins"]),
        jsonToVector(meta["xyz"]["maxs"]),
        device_type);

    torch::Tensor features_dc;
    torch::Tensor opacity;
    torch::Tensor scaling;
    if (attr_u16) {
        features_dc = dequantize2D<uint16_t>(
            readBinary<uint16_t>(result_dir / "f_dc.bin", static_cast<std::size_t>(num_points * 3)),
            num_points,
            3,
            jsonToVector(meta["f_dc"]["mins"]),
            jsonToVector(meta["f_dc"]["maxs"]),
            device_type).view({num_points, 1, 3});
        opacity = dequantize2D<uint16_t>(
            readBinary<uint16_t>(result_dir / "opacity.bin", static_cast<std::size_t>(num_points)),
            num_points,
            1,
            jsonToVector(meta["opacity"]["mins"]),
            jsonToVector(meta["opacity"]["maxs"]),
            device_type);
        scaling = dequantize2D<uint16_t>(
            readBinary<uint16_t>(result_dir / "scaling.bin", static_cast<std::size_t>(num_points * 3)),
            num_points,
            3,
            jsonToVector(meta["scaling"]["mins"]),
            jsonToVector(meta["scaling"]["maxs"]),
            device_type);
    }
    else {
        features_dc = dequantize2D<uint8_t>(
            readBinary<uint8_t>(result_dir / "f_dc.bin", static_cast<std::size_t>(num_points * 3)),
            num_points,
            3,
            jsonToVector(meta["f_dc"]["mins"]),
            jsonToVector(meta["f_dc"]["maxs"]),
            device_type).view({num_points, 1, 3});
        opacity = dequantize2D<uint8_t>(
            readBinary<uint8_t>(result_dir / "opacity.bin", static_cast<std::size_t>(num_points)),
            num_points,
            1,
            jsonToVector(meta["opacity"]["mins"]),
            jsonToVector(meta["opacity"]["maxs"]),
            device_type);
        scaling = dequantize2D<uint8_t>(
            readBinary<uint8_t>(result_dir / "scaling.bin", static_cast<std::size_t>(num_points * 3)),
            num_points,
            3,
            jsonToVector(meta["scaling"]["mins"]),
            jsonToVector(meta["scaling"]["maxs"]),
            device_type);
    }

    torch::Tensor rotation;
    if (rot_u16) {
        rotation = dequantize2D<uint16_t>(
            readBinary<uint16_t>(result_dir / "rotation.bin", static_cast<std::size_t>(num_points * 4)),
            num_points,
            4,
            jsonToVector(meta["rotation"]["mins"]),
            jsonToVector(meta["rotation"]["maxs"]),
            device_type);
    }
    else {
        rotation = dequantize2D<uint8_t>(
            readBinary<uint8_t>(result_dir / "rotation.bin", static_cast<std::size_t>(num_points * 4)),
            num_points,
            4,
            jsonToVector(meta["rotation"]["mins"]),
            jsonToVector(meta["rotation"]["maxs"]),
            device_type);
    }

    const std::size_t rest_payload_values = static_cast<std::size_t>(meta["rest_payload_values"].asUInt64());
    std::vector<float> rest_payload;
    if (attr_u16) {
        rest_payload = dequantize1D<uint16_t>(
            readBinary<uint16_t>(result_dir / "f_rest.bin", rest_payload_values),
            meta["f_rest"]["min"].asFloat(),
            meta["f_rest"]["max"].asFloat());
    }
    else {
        rest_payload = dequantize1D<uint8_t>(
            readBinary<uint8_t>(result_dir / "f_rest.bin", rest_payload_values),
            meta["f_rest"]["min"].asFloat(),
            meta["f_rest"]["max"].asFloat());
    }

    DecodedGaussianTensors decoded;
    decoded.max_sh_degree = max_sh_degree;
    decoded.active_sh_degree = active_sh_degree;
    decoded.xyz = xyz;
    decoded.features_dc = features_dc;
    decoded.features_rest = restoreRestPayload(rest_payload, sh_levels, num_points, max_sh_degree, device_type);
    decoded.opacity = opacity;
    decoded.scaling = scaling;
    decoded.rotation = rotation;
    decoded.sh_levels = sh_levels;
    return decoded;
}
