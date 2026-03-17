/**
 * This file is part of OmniGS
 */

#pragma once

#include <filesystem>

#include <json/json.h>
#include <torch/torch.h>

namespace geometry_codec
{

struct GeometryCodecOptions
{
    int quant_bits = 16;
};

void encodeMortonDelta(
    const torch::Tensor& xyz,
    const std::filesystem::path& path,
    Json::Value& meta,
    const GeometryCodecOptions& options);

torch::Tensor decodeMortonDelta(
    const std::filesystem::path& path,
    const Json::Value& meta,
    torch::DeviceType device_type);

} // namespace geometry_codec
