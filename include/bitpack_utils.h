/**
 * This file is part of OmniGS
 */

#pragma once

#include <filesystem>
#include <cstdint>
#include <vector>

#include <torch/torch.h>

namespace bitpack_utils
{

std::uint8_t minimumBitsForValue(std::uint32_t max_value);

std::vector<std::uint8_t> packUnsignedValues(
    const std::vector<std::uint32_t>& values,
    std::uint8_t bits_per_value);

std::vector<std::uint32_t> unpackUnsignedValues(
    const std::vector<std::uint8_t>& packed,
    std::size_t value_count,
    std::uint8_t bits_per_value);

void writePackedUnsignedTensor(
    const std::filesystem::path& path,
    const torch::Tensor& values,
    std::uint8_t bits_per_value);

torch::Tensor readPackedUnsignedTensor(
    const std::filesystem::path& path,
    std::size_t value_count,
    std::uint8_t bits_per_value,
    torch::DeviceType device_type);

} // namespace bitpack_utils
