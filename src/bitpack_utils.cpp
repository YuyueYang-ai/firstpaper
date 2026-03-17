/**
 * This file is part of OmniGS
 */

#include "include/bitpack_utils.h"

#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace bitpack_utils
{

std::uint8_t minimumBitsForValue(std::uint32_t max_value)
{
    std::uint8_t bits = 0;
    while ((bits < 32u) && ((max_value >> bits) != 0u))
        ++bits;
    return std::max<std::uint8_t>(1u, bits);
}

std::vector<std::uint8_t> packUnsignedValues(
    const std::vector<std::uint32_t>& values,
    std::uint8_t bits_per_value)
{
    if (bits_per_value == 0u || bits_per_value > 32u)
        throw std::runtime_error("packUnsignedValues: invalid bit width.");

    const std::size_t total_bits = values.size() * static_cast<std::size_t>(bits_per_value);
    std::vector<std::uint8_t> packed((total_bits + 7u) / 8u, 0u);
    std::size_t bit_offset = 0;
    const std::uint64_t mask = bits_per_value == 32u
        ? std::numeric_limits<std::uint32_t>::max()
        : ((std::uint64_t{1} << bits_per_value) - 1u);

    for (const auto value : values) {
        std::uint64_t bits = static_cast<std::uint64_t>(value) & mask;
        for (std::uint8_t bit = 0; bit < bits_per_value; ++bit, ++bit_offset) {
            if (((bits >> bit) & 1u) == 0u)
                continue;
            packed[bit_offset / 8u] |= static_cast<std::uint8_t>(1u << (bit_offset % 8u));
        }
    }
    return packed;
}

std::vector<std::uint32_t> unpackUnsignedValues(
    const std::vector<std::uint8_t>& packed,
    std::size_t value_count,
    std::uint8_t bits_per_value)
{
    if (bits_per_value == 0u || bits_per_value > 32u)
        throw std::runtime_error("unpackUnsignedValues: invalid bit width.");

    std::vector<std::uint32_t> values(value_count, 0u);
    std::size_t bit_offset = 0;
    for (std::size_t idx = 0; idx < value_count; ++idx) {
        std::uint32_t value = 0u;
        for (std::uint8_t bit = 0; bit < bits_per_value; ++bit, ++bit_offset) {
            const std::size_t byte_idx = bit_offset / 8u;
            if (byte_idx >= packed.size())
                throw std::runtime_error("unpackUnsignedValues: packed buffer truncated.");
            const std::uint8_t bit_value = static_cast<std::uint8_t>((packed[byte_idx] >> (bit_offset % 8u)) & 1u);
            value |= static_cast<std::uint32_t>(bit_value) << bit;
        }
        values[idx] = value;
    }
    return values;
}

void writePackedUnsignedTensor(
    const std::filesystem::path& path,
    const torch::Tensor& values,
    std::uint8_t bits_per_value)
{
    auto cpu_values = values.detach().contiguous().view({-1}).to(torch::kCPU, torch::kInt32);
    std::vector<std::uint32_t> host_values(static_cast<std::size_t>(cpu_values.numel()), 0u);
    const auto* src_ptr = cpu_values.data_ptr<std::int32_t>();
    for (std::size_t idx = 0; idx < host_values.size(); ++idx) {
        if (src_ptr[idx] < 0)
            throw std::runtime_error("writePackedUnsignedTensor: negative values are unsupported.");
        host_values[idx] = static_cast<std::uint32_t>(src_ptr[idx]);
    }
    const auto packed = packUnsignedValues(host_values, bits_per_value);

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open packed tensor file at " + path.string());
    if (!packed.empty())
        out.write(reinterpret_cast<const char*>(packed.data()), static_cast<std::streamsize>(packed.size()));
}

torch::Tensor readPackedUnsignedTensor(
    const std::filesystem::path& path,
    std::size_t value_count,
    std::uint8_t bits_per_value,
    torch::DeviceType device_type)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open())
        throw std::runtime_error("Cannot open packed tensor file at " + path.string());
    const auto file_size = static_cast<std::size_t>(in.tellg());
    in.seekg(0, std::ios::beg);

    std::vector<std::uint8_t> packed(file_size, 0u);
    if (file_size > 0)
        in.read(reinterpret_cast<char*>(packed.data()), static_cast<std::streamsize>(file_size));
    const auto unpacked = unpackUnsignedValues(packed, value_count, bits_per_value);

    auto tensor = torch::empty(
        {static_cast<int64_t>(value_count)},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto* dst_ptr = tensor.data_ptr<std::int32_t>();
    for (std::size_t idx = 0; idx < unpacked.size(); ++idx)
        dst_ptr[idx] = static_cast<std::int32_t>(unpacked[idx]);
    return tensor.to(device_type);
}

} // namespace bitpack_utils
