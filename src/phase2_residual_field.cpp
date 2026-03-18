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

#include "include/phase2_residual_field.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <lzma.h>
#include <zlib.h>

#include <json/json.h>
#include <torch/torch.h>

#include "include/attribute_sort.h"
#include "include/bitpack_utils.h"
#include "include/gaussian_codec.h"
#include "include/general_utils.h"
#include "include/geometry_codec.h"
#include "include/locality_codec.h"
#include "include/sh_bandwidth.h"
#include "include/phase2_hybrid_selector.h"
#include "include/sh_bandwidth.h"

namespace
{

constexpr float kPi = 3.14159265358979323846f;

void ensureDirectory(const std::filesystem::path& path)
{
    if (!path.empty() && !std::filesystem::exists(path) && !std::filesystem::create_directories(path))
        throw std::runtime_error("Cannot create phase2 directory at " + path.string());
}

Json::Value tensorShapeJson(const torch::Tensor& tensor)
{
    Json::Value shape(Json::arrayValue);
    for (const auto dim : tensor.sizes())
        shape.append(Json::Value::Int64(dim));
    return shape;
}

Json::Value hybridSelectorSummaryJson(const phase2_hybrid_selector::HybridSelectionResult& selection)
{
    Json::Value root;
    root["enabled"] = selection.enabled;
    root["num_hard_blocks"] = Json::Value::Int64(selection.num_hard_blocks);
    root["num_hard_points"] = Json::Value::Int64(selection.num_hard_points);
    root["realized_hard_point_ratio"] = selection.realized_hard_point_ratio;
    root["realized_hard_block_ratio"] = selection.realized_hard_block_ratio;
    root["mean_score_hard"] = selection.mean_score_hard;
    root["mean_score_easy"] = selection.mean_score_easy;
    root["mean_block_mse_hard"] = selection.mean_block_mse_hard;
    root["mean_block_mse_easy"] = selection.mean_block_mse_easy;
    root["mean_sh_level_hard"] = selection.mean_sh_level_hard;
    root["mean_sh_level_easy"] = selection.mean_sh_level_easy;
    root["mean_block_explicit_bpp_hard"] = selection.mean_block_explicit_bpp_hard;
    root["mean_block_explicit_bpp_easy"] = selection.mean_block_explicit_bpp_easy;
    return root;
}

std::vector<int64_t> tensorShapeFromJson(const Json::Value& shape_json)
{
    std::vector<int64_t> shape;
    if (!shape_json.isArray())
        return shape;
    shape.reserve(shape_json.size());
    for (const auto& dim : shape_json)
        shape.push_back(dim.asInt64());
    return shape;
}

std::size_t elementCount(const std::vector<int64_t>& shape)
{
    if (shape.empty())
        return 0;
    return static_cast<std::size_t>(std::accumulate(
        shape.begin(),
        shape.end(),
        int64_t{1},
        [](int64_t lhs, int64_t rhs) { return lhs * rhs; }));
}

template <typename T>
void writeTensorBinary(const std::filesystem::path& path, const torch::Tensor& tensor)
{
    auto cpu_tensor = tensor.detach().contiguous().to(torch::kCPU);
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open phase2 tensor file at " + path.string());
    out.write(reinterpret_cast<const char*>(cpu_tensor.data_ptr<T>()), static_cast<std::streamsize>(cpu_tensor.nbytes()));
}

template <typename T>
torch::Tensor readTensorBinary(
    const std::filesystem::path& path,
    const std::vector<int64_t>& shape,
    torch::ScalarType dtype,
    torch::DeviceType device_type)
{
    const auto count = elementCount(shape);
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open phase2 tensor file at " + path.string());
    std::vector<T> buffer(count);
    if (count > 0)
        in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(count * sizeof(T)));
    auto tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
    if (count > 0)
        std::memcpy(tensor.data_ptr<T>(), buffer.data(), count * sizeof(T));
    return tensor.to(device_type);
}

Json::Value readMetadataJson(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in.is_open())
        throw std::runtime_error("Cannot open phase2 metadata file at " + path.string());
    Json::CharReaderBuilder builder;
    Json::Value root;
    std::string errs;
    if (!Json::parseFromStream(builder, in, &root, &errs))
        throw std::runtime_error("Cannot parse phase2 metadata at " + path.string() + ": " + errs);
    return root;
}

void writeMetadata(const std::filesystem::path& path, const Json::Value& root)
{
    std::ofstream out(path);
    if (!out.is_open())
        throw std::runtime_error("Cannot open phase2 metadata file at " + path.string());
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(root, &out);
}

Phase2ResidualFieldTrainOptions trainOptionsFromJson(const Json::Value& root)
{
    Phase2ResidualFieldTrainOptions options;
    if (root.isMember("num_fourier_frequencies"))
        options.num_fourier_frequencies = root["num_fourier_frequencies"].asInt();
    if (root.isMember("use_hashgrid_encoder"))
        options.use_hashgrid_encoder = root["use_hashgrid_encoder"].asBool();
    if (root.isMember("hashgrid_num_levels"))
        options.hashgrid_num_levels = root["hashgrid_num_levels"].asInt();
    if (root.isMember("hashgrid_features_per_level"))
        options.hashgrid_features_per_level = root["hashgrid_features_per_level"].asInt();
    if (root.isMember("hashgrid_log2_hashmap_size"))
        options.hashgrid_log2_hashmap_size = root["hashgrid_log2_hashmap_size"].asInt();
    if (root.isMember("hashgrid_base_resolution"))
        options.hashgrid_base_resolution = root["hashgrid_base_resolution"].asInt();
    if (root.isMember("hashgrid_per_level_scale"))
        options.hashgrid_per_level_scale = root["hashgrid_per_level_scale"].asFloat();
    if (root.isMember("hidden_dim"))
        options.hidden_dim = root["hidden_dim"].asInt();
    if (root.isMember("num_hidden_layers"))
        options.num_hidden_layers = root["num_hidden_layers"].asInt();
    if (root.isMember("batch_size"))
        options.batch_size = root["batch_size"].asInt();
    if (root.isMember("max_steps"))
        options.max_steps = root["max_steps"].asInt();
    if (root.isMember("log_interval"))
        options.log_interval = root["log_interval"].asInt();
    if (root.isMember("eval_interval"))
        options.eval_interval = root["eval_interval"].asInt();
    if (root.isMember("learning_rate"))
        options.learning_rate = root["learning_rate"].asFloat();
    if (root.isMember("weight_decay"))
        options.weight_decay = root["weight_decay"].asFloat();
    if (root.isMember("include_features_dc"))
        options.include_features_dc = root["include_features_dc"].asBool();
    if (root.isMember("include_opacity"))
        options.include_opacity = root["include_opacity"].asBool();
    if (root.isMember("include_scaling"))
        options.include_scaling = root["include_scaling"].asBool();
    if (root.isMember("include_rotation"))
        options.include_rotation = root["include_rotation"].asBool();
    if (root.isMember("predict_opacity"))
        options.predict_opacity = root["predict_opacity"].asBool();
    if (root.isMember("predict_scaling"))
        options.predict_scaling = root["predict_scaling"].asBool();
    if (root.isMember("predict_rotation"))
        options.predict_rotation = root["predict_rotation"].asBool();
    if (root.isMember("block_embedding_dim"))
        options.block_embedding_dim = root["block_embedding_dim"].asInt();
    if (root.isMember("hybrid_hard_only"))
        options.hybrid_hard_only = root["hybrid_hard_only"].asBool();
    if (root.isMember("hybrid_override_rest_only"))
        options.hybrid_override_rest_only = root["hybrid_override_rest_only"].asBool();
    if (root.isMember("hybrid_easy_export_sh_drop"))
        options.hybrid_easy_export_sh_drop = root["hybrid_easy_export_sh_drop"].asBool();
    if (root.isMember("hybrid_easy_export_sh_preserve_blocks"))
        options.hybrid_easy_export_sh_preserve_blocks = root["hybrid_easy_export_sh_preserve_blocks"].asBool();
    if (root.isMember("hybrid_easy_export_sh_energy_keep_ratio"))
        options.hybrid_easy_export_sh_energy_keep_ratio = root["hybrid_easy_export_sh_energy_keep_ratio"].asFloat();
    if (root.isMember("hybrid_easy_export_sh_min_opacity"))
        options.hybrid_easy_export_sh_min_opacity = root["hybrid_easy_export_sh_min_opacity"].asFloat();
    if (root.isMember("hybrid_easy_export_sh_min_level"))
        options.hybrid_easy_export_sh_min_level = root["hybrid_easy_export_sh_min_level"].asInt();
    if (root.isMember("save_decoded_compact"))
        options.save_decoded_compact = root["save_decoded_compact"].asBool();
    if (root.isMember("save_phase2_compact"))
        options.save_phase2_compact = root["save_phase2_compact"].asBool();
    if (root.isMember("decoded_xyz_quant_bits"))
        options.decoded_xyz_quant_bits = root["decoded_xyz_quant_bits"].asInt();
    if (root.isMember("decoded_attribute_quant_bits"))
        options.decoded_attribute_quant_bits = root["decoded_attribute_quant_bits"].asInt();
    if (root.isMember("decoded_rotation_quant_bits"))
        options.decoded_rotation_quant_bits = root["decoded_rotation_quant_bits"].asInt();
    if (root.isMember("phase2_compact_opacity_quant_bits"))
        options.phase2_compact_opacity_quant_bits = root["phase2_compact_opacity_quant_bits"].asInt();
    if (root.isMember("phase2_compact_scaling_quant_bits"))
        options.phase2_compact_scaling_quant_bits = root["phase2_compact_scaling_quant_bits"].asInt();
    if (root.isMember("phase2_compact_pack_sh_levels"))
        options.phase2_compact_pack_sh_levels = root["phase2_compact_pack_sh_levels"].asBool();
    if (root.isMember("phase2_compact_fdc_quant_bits"))
        options.phase2_compact_fdc_quant_bits = root["phase2_compact_fdc_quant_bits"].asInt();
    if (root.isMember("phase2_compact_easy_rest_base_quant_bits"))
        options.phase2_compact_easy_rest_base_quant_bits = root["phase2_compact_easy_rest_base_quant_bits"].asInt();
    if (root.isMember("phase2_compact_easy_rest_scale_quant_bits"))
        options.phase2_compact_easy_rest_scale_quant_bits = root["phase2_compact_easy_rest_scale_quant_bits"].asInt();
    if (root.isMember("phase2_compact_easy_rest_int2_rel_mse_threshold"))
        options.phase2_compact_easy_rest_int2_rel_mse_threshold = root["phase2_compact_easy_rest_int2_rel_mse_threshold"].asFloat();
    if (root.isMember("phase2_compact_use_geometry_codec"))
        options.phase2_compact_use_geometry_codec = root["phase2_compact_use_geometry_codec"].asBool();
    if (root.isMember("phase2_compact_geometry_quant_bits"))
        options.phase2_compact_geometry_quant_bits = root["phase2_compact_geometry_quant_bits"].asInt();
    if (root.isMember("phase2_compact_store_field_fp16"))
        options.phase2_compact_store_field_fp16 = root["phase2_compact_store_field_fp16"].asBool();
    if (root.isMember("phase2_compact_easy_rest_zlib"))
        options.phase2_compact_easy_rest_zlib = root["phase2_compact_easy_rest_zlib"].asBool();
    if (root.isMember("phase2_compact_easy_rest_zlib_level"))
        options.phase2_compact_easy_rest_zlib_level = root["phase2_compact_easy_rest_zlib_level"].asInt();
    if (root.isMember("phase2_compact_quantized_tensor_zlib"))
        options.phase2_compact_quantized_tensor_zlib = root["phase2_compact_quantized_tensor_zlib"].asBool();
    if (root.isMember("phase2_compact_quantized_tensor_zlib_level"))
        options.phase2_compact_quantized_tensor_zlib_level = root["phase2_compact_quantized_tensor_zlib_level"].asInt();
    if (root.isMember("phase2_compact_geometry_zlib"))
        options.phase2_compact_geometry_zlib = root["phase2_compact_geometry_zlib"].asBool();
    if (root.isMember("phase2_compact_geometry_zlib_level"))
        options.phase2_compact_geometry_zlib_level = root["phase2_compact_geometry_zlib_level"].asInt();
    if (root.isMember("phase2_compact_field_zlib"))
        options.phase2_compact_field_zlib = root["phase2_compact_field_zlib"].asBool();
    if (root.isMember("phase2_compact_field_zlib_level"))
        options.phase2_compact_field_zlib_level = root["phase2_compact_field_zlib_level"].asInt();
    if (root.isMember("phase2_compact_use_xz"))
        options.phase2_compact_use_xz = root["phase2_compact_use_xz"].asBool();
    return options;
}

struct PredictionSlices
{
    torch::Tensor rest_residual;
    torch::Tensor opacity_residual;
    torch::Tensor scaling_residual;
    torch::Tensor rotation_residual;
};

PredictionSlices splitPrediction(
    const torch::Tensor& prediction,
    int max_sh_degree,
    const Phase2ResidualFieldTrainOptions& options)
{
    PredictionSlices slices;
    const int rest_dim = (((max_sh_degree + 1) * (max_sh_degree + 1)) - 1) * 3;
    int offset = 0;
    slices.rest_residual = prediction.index({torch::indexing::Slice(), torch::indexing::Slice(offset, offset + rest_dim)});
    offset += rest_dim;
    if (options.predict_opacity) {
        slices.opacity_residual = prediction.index({torch::indexing::Slice(), torch::indexing::Slice(offset, offset + 1)});
        offset += 1;
    }
    if (options.predict_scaling) {
        slices.scaling_residual = prediction.index({torch::indexing::Slice(), torch::indexing::Slice(offset, offset + 3)});
        offset += 3;
    }
    if (options.predict_rotation) {
        slices.rotation_residual = prediction.index({torch::indexing::Slice(), torch::indexing::Slice(offset, offset + 4)});
        offset += 4;
    }
    return slices;
}

torch::Tensor flattenTensor(const torch::Tensor& tensor)
{
    if (!tensor.defined() || tensor.numel() == 0)
        return tensor;
    return tensor.view({tensor.size(0), -1});
}

torch::Tensor maybeNormalizeQuaternionRows(const torch::Tensor& rotation)
{
    if (!rotation.defined() || rotation.numel() == 0)
        return rotation;
    auto norm = torch::clamp_min(torch::norm(rotation, 2, 1, true), 1e-8f);
    return rotation / norm;
}

torch::Tensor sanitizeTensorFinite(
    const torch::Tensor& tensor,
    float nan_value = 0.0f,
    float posinf_value = 0.0f,
    float neginf_value = 0.0f)
{
    return torch::nan_to_num(tensor.to(torch::kFloat32), nan_value, posinf_value, neginf_value);
}

torch::Tensor sanitizeOpacityLogits(const torch::Tensor& opacity)
{
    return torch::nan_to_num(opacity.to(torch::kFloat32), 0.0f, 20.0f, -20.0f);
}

torch::Tensor buildPerPointBase(
    const torch::Tensor& target,
    const torch::Tensor& block_ids,
    int64_t num_blocks,
    bool normalize_quaternion = false)
{
    auto block_means = locality_codec::computeBlockMeans(target, block_ids, num_blocks).to(torch::kFloat32);
    if (normalize_quaternion)
        block_means = maybeNormalizeQuaternionRows(flattenTensor(block_means)).view_as(block_means);
    auto expanded = locality_codec::expandBlockMeans(block_means, block_ids).to(torch::kFloat32);
    if (normalize_quaternion)
        expanded = maybeNormalizeQuaternionRows(flattenTensor(expanded)).view_as(expanded);
    return expanded;
}

torch::Tensor blockMeansFromPerPointBase(
    const torch::Tensor& base_per_point,
    const torch::Tensor& block_ids,
    int64_t num_blocks,
    bool normalize_quaternion = false)
{
    auto block_means = locality_codec::computeBlockMeans(base_per_point, block_ids, num_blocks).to(torch::kFloat32);
    if (normalize_quaternion)
        block_means = maybeNormalizeQuaternionRows(flattenTensor(block_means)).view_as(block_means);
    return block_means;
}

std::pair<torch::Tensor, int64_t> remapBlockIdsContiguous(const torch::Tensor& block_ids)
{
    if (!block_ids.defined() || block_ids.numel() == 0)
        return {torch::zeros({0}, torch::TensorOptions().dtype(torch::kLong).device(block_ids.device())), 0};

    auto block_ids_cpu = block_ids.detach().contiguous().to(torch::kCPU, torch::kLong);
    auto remapped_cpu = torch::zeros_like(block_ids_cpu);
    const auto* src_ptr = block_ids_cpu.data_ptr<int64_t>();
    auto* dst_ptr = remapped_cpu.data_ptr<int64_t>();

    int64_t current = std::numeric_limits<int64_t>::min();
    int64_t remapped = -1;
    for (int64_t idx = 0; idx < block_ids_cpu.size(0); ++idx) {
        if (src_ptr[idx] != current) {
            current = src_ptr[idx];
            ++remapped;
        }
        dst_ptr[idx] = remapped;
    }
    return {remapped_cpu.to(block_ids.device()), remapped + 1};
}

torch::Tensor estimateBlockwiseExportLevels(
    const torch::Tensor& features_rest,
    const torch::Tensor& opacity_logits,
    const torch::Tensor& block_ids,
    int64_t num_blocks,
    int max_sh_degree,
    float energy_keep_ratio,
    float min_opacity,
    int min_level)
{
    if (!features_rest.defined() || features_rest.numel() == 0 || !block_ids.defined() || block_ids.numel() == 0 || num_blocks <= 0)
        return torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt32).device(features_rest.device()));

    auto block_rest = locality_codec::computeBlockMeans(features_rest.to(torch::kFloat32), block_ids, num_blocks).to(torch::kFloat32);
    auto block_opacity = locality_codec::computeBlockMeans(
        torch::sigmoid(opacity_logits.view({opacity_logits.size(0), -1}).to(torch::kFloat32)),
        block_ids,
        num_blocks).to(torch::kFloat32);
    auto block_levels = sh_bandwidth::estimateLevels(
        block_rest,
        block_opacity,
        max_sh_degree,
        energy_keep_ratio,
        min_opacity,
        min_level).to(torch::kInt32);

    auto expanded = locality_codec::expandBlockMeans(
        block_levels.to(torch::kFloat32).view({num_blocks, 1}),
        block_ids).view({-1}).round().to(torch::kInt32);
    return expanded.to(features_rest.device());
}

void writeBinaryBytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open binary output at " + path.string());
    if (!bytes.empty())
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

std::vector<std::uint8_t> readBinaryBytes(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open())
        throw std::runtime_error("Cannot open binary input at " + path.string());
    const auto size = static_cast<std::size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(size, 0u);
    if (size > 0)
        in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    return bytes;
}

void writeRestBlockInfoBinary(
    const std::filesystem::path& path,
    const std::vector<locality_codec::RestBlockInfo>& blocks)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open rest-block binary output at " + path.string());
    for (const auto& block : blocks) {
        out.put(static_cast<char>(block.sh_level));
        const std::uint16_t point_count = block.point_count;
        out.put(static_cast<char>(point_count & 0xFFu));
        out.put(static_cast<char>((point_count >> 8u) & 0xFFu));
        out.put(static_cast<char>(block.residual_bits));
    }
}

std::vector<locality_codec::RestBlockInfo> readRestBlockInfoBinary(
    const std::filesystem::path& path,
    std::size_t count)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open rest-block binary input at " + path.string());
    std::vector<locality_codec::RestBlockInfo> blocks(count);
    for (std::size_t idx = 0; idx < count; ++idx) {
        int ch0 = in.get();
        int ch1 = in.get();
        int ch2 = in.get();
        int ch3 = in.get();
        if (ch0 == EOF || ch1 == EOF || ch2 == EOF || ch3 == EOF)
            throw std::runtime_error("Rest-block binary is truncated at " + path.string());
        blocks[idx].sh_level = static_cast<std::uint8_t>(ch0);
        blocks[idx].point_count = static_cast<std::uint16_t>(static_cast<std::uint16_t>(ch1) | (static_cast<std::uint16_t>(ch2) << 8u));
        blocks[idx].residual_bits = static_cast<std::uint8_t>(ch3);
    }
    return blocks;
}

void writeHalfVectorBinary(const std::filesystem::path& path, const std::vector<c10::Half>& values)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open half-vector output at " + path.string());
    if (!values.empty())
        out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(c10::Half)));
}

std::vector<c10::Half> readHalfVectorBinary(const std::filesystem::path& path, std::size_t count)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open half-vector input at " + path.string());
    std::vector<c10::Half> values(count);
    if (count > 0)
        in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(c10::Half)));
    return values;
}

torch::Tensor halfVectorToTensor(const std::vector<c10::Half>& values)
{
    auto tensor = torch::empty(
        {static_cast<int64_t>(values.size()), 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    float* dst_ptr = tensor.data_ptr<float>();
    for (std::size_t idx = 0; idx < values.size(); ++idx)
        dst_ptr[idx] = static_cast<float>(values[idx]);
    return tensor;
}

std::vector<c10::Half> tensorToHalfVector(const torch::Tensor& tensor)
{
    auto flat = tensor.detach().contiguous().to(torch::kCPU, torch::kFloat32).view({-1});
    const float* src_ptr = flat.data_ptr<float>();
    std::vector<c10::Half> values(static_cast<std::size_t>(flat.numel()));
    for (int64_t idx = 0; idx < flat.numel(); ++idx)
        values[static_cast<std::size_t>(idx)] = c10::Half(src_ptr[idx]);
    return values;
}

std::vector<std::uint8_t> zlibCompressBytes(const std::vector<std::uint8_t>& input, int level)
{
    if (input.empty())
        return {};
    uLongf compressed_bound = compressBound(static_cast<uLong>(input.size()));
    std::vector<std::uint8_t> compressed(static_cast<std::size_t>(compressed_bound));
    const int code = compress2(
        reinterpret_cast<Bytef*>(compressed.data()),
        &compressed_bound,
        reinterpret_cast<const Bytef*>(input.data()),
        static_cast<uLong>(input.size()),
        std::min(9, std::max(0, level)));
    if (code != Z_OK)
        throw std::runtime_error("zlib compression failed for encoded rest payload.");
    compressed.resize(static_cast<std::size_t>(compressed_bound));
    return compressed;
}

std::vector<std::uint8_t> zlibDecompressBytes(const std::vector<std::uint8_t>& input, std::size_t uncompressed_size)
{
    if (uncompressed_size == 0)
        return {};
    std::vector<std::uint8_t> output(uncompressed_size);
    uLongf actual_size = static_cast<uLongf>(uncompressed_size);
    const int code = uncompress(
        reinterpret_cast<Bytef*>(output.data()),
        &actual_size,
        reinterpret_cast<const Bytef*>(input.data()),
        static_cast<uLong>(input.size()));
    if (code != Z_OK)
        throw std::runtime_error("zlib decompression failed for encoded rest payload.");
    output.resize(static_cast<std::size_t>(actual_size));
    return output;
}

std::vector<std::uint8_t> xzCompressBytes(const std::vector<std::uint8_t>& input, int level)
{
    if (input.empty())
        return {};
    std::vector<std::uint8_t> compressed(static_cast<std::size_t>(lzma_stream_buffer_bound(input.size())));
    std::size_t out_pos = 0;
    const std::uint32_t preset = static_cast<std::uint32_t>(std::clamp(level, 0, 9)) | LZMA_PRESET_EXTREME;
    const lzma_ret code = lzma_easy_buffer_encode(
        preset,
        LZMA_CHECK_CRC64,
        nullptr,
        reinterpret_cast<const std::uint8_t*>(input.data()),
        input.size(),
        reinterpret_cast<std::uint8_t*>(compressed.data()),
        &out_pos,
        compressed.size());
    if (code != LZMA_OK)
        throw std::runtime_error("xz compression failed for Phase 2 compact payload.");
    compressed.resize(out_pos);
    return compressed;
}

std::vector<std::uint8_t> xzDecompressBytes(const std::vector<std::uint8_t>& input, std::size_t uncompressed_size)
{
    if (uncompressed_size == 0)
        return {};
    std::vector<std::uint8_t> output(uncompressed_size);
    std::size_t in_pos = 0;
    std::size_t out_pos = 0;
    std::uint64_t memlimit = std::numeric_limits<std::uint64_t>::max();
    const lzma_ret code = lzma_stream_buffer_decode(
        &memlimit,
        0,
        nullptr,
        reinterpret_cast<const std::uint8_t*>(input.data()),
        &in_pos,
        input.size(),
        reinterpret_cast<std::uint8_t*>(output.data()),
        &out_pos,
        output.size());
    if (code != LZMA_OK)
        throw std::runtime_error("xz decompression failed for Phase 2 compact payload.");
    output.resize(out_pos);
    return output;
}

std::string resolveLosslessCodec(bool enabled, bool use_xz)
{
    if (!enabled)
        return "raw";
    return use_xz ? "xz" : "zlib";
}

std::vector<std::uint8_t> compressBytesWithCodec(
    const std::vector<std::uint8_t>& input,
    const std::string& codec,
    int level)
{
    if (codec == "raw")
        return input;
    if (codec == "zlib")
        return zlibCompressBytes(input, level);
    if (codec == "xz")
        return xzCompressBytes(input, level);
    throw std::runtime_error("Unsupported lossless codec: " + codec);
}

std::vector<std::uint8_t> decompressBytesWithCodec(
    const std::vector<std::uint8_t>& input,
    const std::string& codec,
    std::size_t uncompressed_size)
{
    if (codec == "raw")
        return input;
    if (codec == "zlib")
        return zlibDecompressBytes(input, uncompressed_size);
    if (codec == "xz")
        return xzDecompressBytes(input, uncompressed_size);
    throw std::runtime_error("Unsupported lossless codec: " + codec);
}

Json::Value maybeCompressFileInPlace(
    const std::filesystem::path& path,
    const std::string& codec,
    int compression_level)
{
    Json::Value meta;
    const auto original_bytes = readBinaryBytes(path);
    meta["byte_storage"] = codec;
    meta["uncompressed_byte_count"] = Json::Value::UInt64(original_bytes.size());
    if (codec != "raw") {
        const auto compressed = compressBytesWithCodec(original_bytes, codec, compression_level);
        writeBinaryBytes(path, compressed);
        meta["byte_count"] = Json::Value::UInt64(compressed.size());
    }
    else {
        meta["byte_count"] = Json::Value::UInt64(original_bytes.size());
    }
    return meta;
}

std::filesystem::path materializeMaybeCompressedFile(
    const std::filesystem::path& path,
    const Json::Value& storage_meta,
    const std::string& suffix)
{
    const auto storage = storage_meta.get("byte_storage", "raw").asString();
    if (storage == "raw")
        return path;

    const auto compressed = readBinaryBytes(path);
    const auto restored = decompressBytesWithCodec(
        compressed,
        storage,
        static_cast<std::size_t>(storage_meta["uncompressed_byte_count"].asUInt64()));
    const auto temp_path = std::filesystem::temp_directory_path() /
        std::filesystem::path(path.filename().string() + "." + suffix + ".tmp");
    writeBinaryBytes(temp_path, restored);
    return temp_path;
}

void saveQuantizedTensorUint(
    const std::filesystem::path& path,
    const torch::Tensor& tensor,
    int quant_bits,
    Json::Value& meta,
    const std::string& codec,
    int compression_level);

torch::Tensor loadQuantizedTensorUint(
    const std::filesystem::path& path,
    const Json::Value& meta,
    torch::DeviceType device_type);

Json::Value saveEncodedRestPayload(
    const std::filesystem::path& result_dir,
    const std::string& prefix,
    const locality_codec::EncodedRestPayload& encoded,
    const std::string& codec = "raw",
    int compression_level = 6,
    int base_quant_bits = 16,
    int scale_quant_bits = 16)
{
    const auto blocks_path = result_dir / (prefix + "_blocks.bin");
    const auto base_values_path = result_dir / (prefix + "_base_values.bin");
    const auto scale_values_path = result_dir / (prefix + "_scale_values.bin");

    writeRestBlockInfoBinary(blocks_path, encoded.blocks);
    const bool quantize_base_values = base_quant_bits > 0 && base_quant_bits <= 8;
    const bool quantize_scale_values = scale_quant_bits > 0 && scale_quant_bits <= 8;
    Json::Value base_values_meta;
    Json::Value scale_values_meta;
    if (quantize_base_values) {
        saveQuantizedTensorUint(
            base_values_path,
            halfVectorToTensor(encoded.base_values),
            base_quant_bits,
            base_values_meta,
            codec,
            compression_level);
    } else {
        writeHalfVectorBinary(base_values_path, encoded.base_values);
    }
    if (quantize_scale_values) {
        saveQuantizedTensorUint(
            scale_values_path,
            halfVectorToTensor(encoded.scale_values),
            scale_quant_bits,
            scale_values_meta,
            codec,
            compression_level);
    } else {
        writeHalfVectorBinary(scale_values_path, encoded.scale_values);
    }

    const bool use_split_residual_streams =
        !encoded.residual_bytes_int2.empty() || !encoded.residual_bytes_int4.empty() || !encoded.residual_bytes_int8.empty();

    std::vector<std::uint8_t> residual_bytes;
    std::vector<std::uint8_t> residual_bytes_int2;
    std::vector<std::uint8_t> residual_bytes_int4;
    std::vector<std::uint8_t> residual_bytes_int8;
    if (use_split_residual_streams) {
        residual_bytes_int2 = encoded.residual_bytes_int2;
        residual_bytes_int4 = encoded.residual_bytes_int4;
        residual_bytes_int8 = encoded.residual_bytes_int8;
        if (codec != "raw") {
            residual_bytes_int2 = compressBytesWithCodec(encoded.residual_bytes_int2, codec, compression_level);
            residual_bytes_int4 = compressBytesWithCodec(encoded.residual_bytes_int4, codec, compression_level);
            residual_bytes_int8 = compressBytesWithCodec(encoded.residual_bytes_int8, codec, compression_level);
        }
        writeBinaryBytes(result_dir / (prefix + "_residual_int2_bytes.bin"), residual_bytes_int2);
        writeBinaryBytes(result_dir / (prefix + "_residual_int4_bytes.bin"), residual_bytes_int4);
        writeBinaryBytes(result_dir / (prefix + "_residual_int8_bytes.bin"), residual_bytes_int8);
    } else {
        residual_bytes = encoded.residual_bytes;
        if (codec != "raw")
            residual_bytes = compressBytesWithCodec(encoded.residual_bytes, codec, compression_level);
        writeBinaryBytes(result_dir / (prefix + "_residual_bytes.bin"), residual_bytes);
    }

    Json::Value meta;
    meta["blocks_storage"] = "binary_u8_u16_u8";
    meta["block_count"] = Json::Value::UInt64(encoded.blocks.size());
    meta["payload_values"] = Json::Value::UInt64(encoded.payload_values);
    meta["base_value_count"] = Json::Value::UInt64(encoded.base_values.size());
    meta["scale_value_count"] = Json::Value::UInt64(encoded.scale_values.size());
    if (quantize_base_values) {
        meta["base_values_format"] = "packed_uint";
        meta["base_values"] = base_values_meta;
    } else {
        meta["base_values_format"] = "half_raw";
    }
    if (quantize_scale_values) {
        meta["scale_values_format"] = "packed_uint";
        meta["scale_values"] = scale_values_meta;
    } else {
        meta["scale_values_format"] = "half_raw";
    }
    if (!quantize_base_values && codec != "raw") {
        meta["base_values_storage"] = maybeCompressFileInPlace(base_values_path, codec, compression_level);
    }
    if (!quantize_scale_values && codec != "raw") {
        meta["scale_values_storage"] = maybeCompressFileInPlace(scale_values_path, codec, compression_level);
    }
    meta["residual_storage"] = codec;
    meta["residual_layout"] = use_split_residual_streams ? "split_by_bitwidth" : "mixed_single_stream";
    if (use_split_residual_streams) {
        meta["residual_int2_byte_count"] = Json::Value::UInt64(residual_bytes_int2.size());
        meta["residual_int2_uncompressed_byte_count"] = Json::Value::UInt64(encoded.residual_bytes_int2.size());
        meta["residual_int4_byte_count"] = Json::Value::UInt64(residual_bytes_int4.size());
        meta["residual_int4_uncompressed_byte_count"] = Json::Value::UInt64(encoded.residual_bytes_int4.size());
        meta["residual_int8_byte_count"] = Json::Value::UInt64(residual_bytes_int8.size());
        meta["residual_int8_uncompressed_byte_count"] = Json::Value::UInt64(encoded.residual_bytes_int8.size());
    } else {
        meta["residual_byte_count"] = Json::Value::UInt64(residual_bytes.size());
        meta["residual_uncompressed_byte_count"] = Json::Value::UInt64(encoded.residual_bytes.size());
    }
    meta["int2_block_count"] = Json::Value::UInt64(encoded.int2_block_count);
    meta["int4_block_count"] = Json::Value::UInt64(encoded.int4_block_count);
    meta["int8_block_count"] = Json::Value::UInt64(encoded.int8_block_count);
    return meta;
}

locality_codec::EncodedRestPayload loadEncodedRestPayload(
    const std::filesystem::path& result_dir,
    const std::string& prefix,
    const Json::Value& meta)
{
    locality_codec::EncodedRestPayload encoded;
    if (meta.isMember("blocks") && meta["blocks"].isArray()) {
        encoded.blocks.reserve(meta["blocks"].size());
        for (const auto& entry : meta["blocks"]) {
            locality_codec::RestBlockInfo block;
            block.sh_level = static_cast<std::uint8_t>(entry["sh_level"].asUInt());
            block.point_count = static_cast<std::uint16_t>(entry["point_count"].asUInt());
            block.residual_bits = static_cast<std::uint8_t>(entry["residual_bits"].asUInt());
            encoded.blocks.push_back(block);
        }
    }
    else {
        encoded.blocks = readRestBlockInfoBinary(
            result_dir / (prefix + "_blocks.bin"),
            static_cast<std::size_t>(meta["block_count"].asUInt64()));
    }
    encoded.payload_values = static_cast<std::size_t>(meta["payload_values"].asUInt64());
    const auto base_values_format = meta.get("base_values_format", "half_raw").asString();
    if (base_values_format == "packed_uint") {
        encoded.base_values = tensorToHalfVector(loadQuantizedTensorUint(
            result_dir / (prefix + "_base_values.bin"),
            meta["base_values"],
            torch::kCPU));
    } else {
        auto base_values_path = materializeMaybeCompressedFile(
            result_dir / (prefix + "_base_values.bin"),
            meta.get("base_values_storage", Json::Value(Json::nullValue)),
            prefix + "_base_values");
        encoded.base_values = readHalfVectorBinary(
            base_values_path,
            static_cast<std::size_t>(meta["base_value_count"].asUInt64()));
        if (base_values_path != result_dir / (prefix + "_base_values.bin"))
            std::filesystem::remove(base_values_path);
    }

    const auto scale_values_format = meta.get("scale_values_format", "half_raw").asString();
    if (scale_values_format == "packed_uint") {
        encoded.scale_values = tensorToHalfVector(loadQuantizedTensorUint(
            result_dir / (prefix + "_scale_values.bin"),
            meta["scale_values"],
            torch::kCPU));
    } else {
        auto scale_values_path = materializeMaybeCompressedFile(
            result_dir / (prefix + "_scale_values.bin"),
            meta.get("scale_values_storage", Json::Value(Json::nullValue)),
            prefix + "_scale_values");
        encoded.scale_values = readHalfVectorBinary(
            scale_values_path,
            static_cast<std::size_t>(meta["scale_value_count"].asUInt64()));
        if (scale_values_path != result_dir / (prefix + "_scale_values.bin"))
            std::filesystem::remove(scale_values_path);
    }

    const auto residual_storage = meta.get("residual_storage", "raw").asString();
    const auto residual_layout = meta.get("residual_layout", "mixed_single_stream").asString();
    if (residual_layout == "split_by_bitwidth") {
        std::vector<std::uint8_t> residual_bytes_int2;
        if (std::filesystem::exists(result_dir / (prefix + "_residual_int2_bytes.bin")))
            residual_bytes_int2 = readBinaryBytes(result_dir / (prefix + "_residual_int2_bytes.bin"));
        auto residual_bytes_int4 = readBinaryBytes(result_dir / (prefix + "_residual_int4_bytes.bin"));
        auto residual_bytes_int8 = readBinaryBytes(result_dir / (prefix + "_residual_int8_bytes.bin"));
        if (residual_storage != "raw") {
            const auto int2_uncompressed = static_cast<std::size_t>(meta.get("residual_int2_uncompressed_byte_count", 0).asUInt64());
            if (!residual_bytes_int2.empty() || int2_uncompressed > 0) {
                encoded.residual_bytes_int2 = decompressBytesWithCodec(
                    residual_bytes_int2,
                    residual_storage,
                    int2_uncompressed);
            }
            encoded.residual_bytes_int4 = decompressBytesWithCodec(
                residual_bytes_int4,
                residual_storage,
                static_cast<std::size_t>(meta["residual_int4_uncompressed_byte_count"].asUInt64()));
            encoded.residual_bytes_int8 = decompressBytesWithCodec(
                residual_bytes_int8,
                residual_storage,
                static_cast<std::size_t>(meta["residual_int8_uncompressed_byte_count"].asUInt64()));
        } else {
            encoded.residual_bytes_int2 = std::move(residual_bytes_int2);
            encoded.residual_bytes_int4 = std::move(residual_bytes_int4);
            encoded.residual_bytes_int8 = std::move(residual_bytes_int8);
        }
    } else {
        auto residual_bytes = readBinaryBytes(result_dir / (prefix + "_residual_bytes.bin"));
        if (residual_storage != "raw") {
            encoded.residual_bytes = decompressBytesWithCodec(
                residual_bytes,
                residual_storage,
                static_cast<std::size_t>(meta["residual_uncompressed_byte_count"].asUInt64()));
        } else {
            encoded.residual_bytes = std::move(residual_bytes);
        }
    }
    encoded.int2_block_count = static_cast<std::size_t>(meta.get("int2_block_count", 0).asUInt64());
    encoded.int4_block_count = static_cast<std::size_t>(meta.get("int4_block_count", 0).asUInt64());
    encoded.int8_block_count = static_cast<std::size_t>(meta.get("int8_block_count", 0).asUInt64());
    return encoded;
}

torch::Tensor restoreRestPayloadTensor(
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
    const std::int32_t* levels_ptr = levels_cpu.data_ptr<std::int32_t>();

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

torch::Tensor indicesFromPackedFlags(
    const torch::Tensor& flags,
    bool select_true)
{
    auto mask = flags.to(torch::kBool);
    if (!select_true)
        mask = torch::logical_not(mask);
    return torch::nonzero(mask).view({-1});
}

torch::Tensor expandHardBlockBasesToPerPoint(
    const torch::Tensor& hard_block_bases,
    const torch::Tensor& block_ids,
    const torch::Tensor& hard_point_flags,
    const torch::Tensor& hard_block_ids,
    int64_t num_points)
{
    auto block_ids_cpu = block_ids.detach().contiguous().to(torch::kCPU, torch::kLong);
    auto hard_flags_cpu = hard_point_flags.detach().contiguous().to(torch::kCPU, torch::kInt32);
    auto hard_block_ids_cpu = hard_block_ids.detach().contiguous().to(torch::kCPU, torch::kLong);
    auto hard_block_bases_cpu = hard_block_bases.detach().contiguous().to(torch::kCPU, torch::kFloat32);

    const int64_t flattened_dims =
        hard_block_bases_cpu.numel() / std::max<int64_t>(1, hard_block_bases_cpu.size(0));
    auto hard_block_bases_flat = hard_block_bases_cpu.view({hard_block_bases_cpu.size(0), flattened_dims});
    auto expanded = torch::zeros(
        {num_points, flattened_dims},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    std::vector<int64_t> hard_lookup(static_cast<std::size_t>(block_ids_cpu.max().item<int64_t>() + 1), -1);
    const int64_t* hard_block_ids_ptr = hard_block_ids_cpu.data_ptr<int64_t>();
    for (int64_t idx = 0; idx < hard_block_ids_cpu.size(0); ++idx) {
        const int64_t global_block_id = hard_block_ids_ptr[idx];
        if (global_block_id >= 0 && global_block_id < static_cast<int64_t>(hard_lookup.size()))
            hard_lookup[static_cast<std::size_t>(global_block_id)] = idx;
    }

    const int64_t* block_ids_ptr = block_ids_cpu.data_ptr<int64_t>();
    const std::int32_t* hard_flags_ptr = hard_flags_cpu.data_ptr<std::int32_t>();
    const float* hard_base_ptr = hard_block_bases_flat.data_ptr<float>();
    float* expanded_ptr = expanded.data_ptr<float>();
    for (int64_t point_idx = 0; point_idx < num_points; ++point_idx) {
        if (hard_flags_ptr[point_idx] == 0)
            continue;
        const int64_t global_block_id = block_ids_ptr[point_idx];
        if (global_block_id < 0 || global_block_id >= static_cast<int64_t>(hard_lookup.size()))
            continue;
        const int64_t local_block_id = hard_lookup[static_cast<std::size_t>(global_block_id)];
        if (local_block_id < 0)
            continue;
        std::memcpy(
            expanded_ptr + point_idx * flattened_dims,
            hard_base_ptr + local_block_id * flattened_dims,
            static_cast<std::size_t>(flattened_dims) * sizeof(float));
    }

    std::vector<int64_t> output_shape = hard_block_bases_cpu.sizes().vec();
    output_shape[0] = num_points;
    return expanded.view(output_shape).to(block_ids.device());
}

void saveQuantizedTensorUint(
    const std::filesystem::path& path,
    const torch::Tensor& tensor,
    int quant_bits,
    Json::Value& meta,
    const std::string& codec = "raw",
    int compression_level = 6)
{
    auto flat = tensor.detach().contiguous().to(torch::kCPU, torch::kFloat32).view({tensor.size(0), -1});
    const int dims = flat.size(1);
    const int bits = std::clamp(quant_bits, 1, 8);
    const std::uint32_t qmax = (1u << bits) - 1u;
    auto mins = std::get<0>(flat.min(0)).contiguous();
    auto maxs = std::get<0>(flat.max(0)).contiguous();

    Json::Value mins_json(Json::arrayValue);
    Json::Value maxs_json(Json::arrayValue);
    std::vector<std::uint32_t> packed_values(static_cast<std::size_t>(flat.numel()), 0u);

    const float* flat_ptr = flat.data_ptr<float>();
    const float* mins_ptr = mins.data_ptr<float>();
    const float* maxs_ptr = maxs.data_ptr<float>();
    for (int dim = 0; dim < dims; ++dim) {
        mins_json.append(mins_ptr[dim]);
        maxs_json.append(maxs_ptr[dim]);
    }
    for (int64_t row = 0; row < flat.size(0); ++row) {
        for (int dim = 0; dim < dims; ++dim) {
            const float min_v = mins_ptr[dim];
            const float max_v = maxs_ptr[dim];
            const float denom = std::max(max_v - min_v, 1e-8f);
            const float normalized = std::clamp((flat_ptr[row * dims + dim] - min_v) / denom, 0.0f, 1.0f);
            packed_values[static_cast<std::size_t>(row * dims + dim)] =
                static_cast<std::uint32_t>(std::llround(static_cast<double>(normalized) * static_cast<double>(qmax)));
        }
    }

    auto packed_bytes = bitpack_utils::packUnsignedValues(packed_values, static_cast<std::uint8_t>(bits));
    if (codec != "raw")
        packed_bytes = compressBytesWithCodec(packed_bytes, codec, compression_level);
    writeBinaryBytes(path, packed_bytes);

    meta["storage"] = "packed_uint";
    meta["bits"] = bits;
    meta["shape"] = tensorShapeJson(tensor);
    meta["mins"] = mins_json;
    meta["maxs"] = maxs_json;
    meta["byte_storage"] = codec;
    meta["packed_byte_count"] = Json::Value::UInt64(packed_bytes.size());
    meta["packed_uncompressed_byte_count"] =
        Json::Value::UInt64(bitpack_utils::packUnsignedValues(packed_values, static_cast<std::uint8_t>(bits)).size());
}

torch::Tensor loadQuantizedTensorUint(
    const std::filesystem::path& path,
    const Json::Value& meta,
    torch::DeviceType device_type)
{
    const auto shape = tensorShapeFromJson(meta["shape"]);
    const auto count = elementCount(shape);
    const auto dims = shape.empty() ? 0 : static_cast<int>(count / static_cast<std::size_t>(shape[0]));
    const auto bits = static_cast<std::uint8_t>(meta["bits"].asUInt());
    auto packed_bytes = readBinaryBytes(path);
    const auto storage = meta.get("byte_storage", "raw").asString();
    if (storage != "raw") {
        packed_bytes = decompressBytesWithCodec(
            packed_bytes,
            storage,
            static_cast<std::size_t>(meta["packed_uncompressed_byte_count"].asUInt64()));
    }
    const auto unpacked = bitpack_utils::unpackUnsignedValues(packed_bytes, count, bits);
    const std::uint32_t qmax = (1u << bits) - 1u;

    std::vector<float> mins(static_cast<std::size_t>(meta["mins"].size()), 0.0f);
    std::vector<float> maxs(static_cast<std::size_t>(meta["maxs"].size()), 0.0f);
    for (Json::ArrayIndex idx = 0; idx < meta["mins"].size(); ++idx) {
        mins[static_cast<std::size_t>(idx)] = meta["mins"][idx].asFloat();
        maxs[static_cast<std::size_t>(idx)] = meta["maxs"][idx].asFloat();
    }

    auto tensor = torch::empty(shape, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto flat = tensor.view({shape.empty() ? 0 : shape[0], dims});
    float* flat_ptr = flat.data_ptr<float>();
    for (int64_t row = 0; row < flat.size(0); ++row) {
        for (int dim = 0; dim < dims; ++dim) {
            const auto q = unpacked[static_cast<std::size_t>(row * dims + dim)];
            const float min_v = mins[static_cast<std::size_t>(dim)];
            const float max_v = maxs[static_cast<std::size_t>(dim)];
            const float normalized = qmax > 0u ? (static_cast<float>(q) / static_cast<float>(qmax)) : 0.0f;
            flat_ptr[row * dims + dim] = min_v + normalized * (max_v - min_v);
        }
    }
    return tensor.to(device_type);
}

torch::Tensor reorderLeadingDimension(const torch::Tensor& tensor, const torch::Tensor& order)
{
    if (!tensor.defined() || tensor.numel() == 0)
        return tensor;
    return tensor.index_select(0, order);
}

torch::Tensor normalizeFromBounds(
    const torch::Tensor& xyz,
    const Json::Value& bbox_min_json,
    const Json::Value& bbox_max_json)
{
    std::vector<float> bbox_min(3, 0.0f);
    std::vector<float> bbox_max(3, 0.0f);
    for (int axis = 0; axis < 3; ++axis) {
        bbox_min[axis] = bbox_min_json[axis].asFloat();
        bbox_max[axis] = bbox_max_json[axis].asFloat();
    }
    auto bbox_min_tensor = torch::tensor(bbox_min, torch::TensorOptions().dtype(torch::kFloat32).device(xyz.device()));
    auto bbox_max_tensor = torch::tensor(bbox_max, torch::TensorOptions().dtype(torch::kFloat32).device(xyz.device()));
    auto bbox_range_tensor = torch::clamp_min(bbox_max_tensor - bbox_min_tensor, 1e-8f);
    return ((xyz - bbox_min_tensor) / bbox_range_tensor) * 2.0f - 1.0f;
}

torch::Tensor buildRestMask(const torch::Tensor& sh_levels, int max_sh_degree)
{
    const auto num_points = sh_levels.size(0);
    const auto rest_channels = (max_sh_degree + 1) * (max_sh_degree + 1) - 1;
    auto ones = torch::ones(
        {num_points, rest_channels, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(sh_levels.device()));
    return sh_bandwidth::applyLevelsToFeaturesRest(ones, sh_levels.to(torch::kInt32), max_sh_degree);
}

torch::Tensor phase2InputOpacity(
    const phase2_residual_field::FrozenResidualFieldPackage& frozen,
    const Phase2ResidualFieldTrainOptions& options)
{
    return (options.predict_opacity && frozen.opacity_base.defined())
        ? frozen.opacity_base
        : frozen.opacity;
}

torch::Tensor phase2InputScaling(
    const phase2_residual_field::FrozenResidualFieldPackage& frozen,
    const Phase2ResidualFieldTrainOptions& options)
{
    return (options.predict_scaling && frozen.scaling_base.defined())
        ? frozen.scaling_base
        : frozen.scaling;
}

torch::Tensor phase2InputRotation(
    const phase2_residual_field::FrozenResidualFieldPackage& frozen,
    const Phase2ResidualFieldTrainOptions& options)
{
    return (options.predict_rotation && frozen.rotation_base.defined())
        ? frozen.rotation_base
        : frozen.rotation;
}

torch::Tensor buildPredictionMask(
    const phase2_residual_field::FrozenResidualFieldPackage& frozen,
    const Phase2ResidualFieldTrainOptions& options)
{
    auto point_mask = torch::ones(
        {frozen.xyz.size(0), 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(frozen.sh_levels.device()));
    if (options.hybrid_hard_only && frozen.hard_point_flags.defined() && frozen.hard_point_flags.numel() == frozen.xyz.size(0))
        point_mask = frozen.hard_point_flags.to(torch::kFloat32).view({frozen.xyz.size(0), 1});

    auto rest_mask = buildRestMask(frozen.sh_levels, frozen.max_sh_degree).view({frozen.xyz.size(0), -1});
    rest_mask = rest_mask * point_mask;
    std::vector<torch::Tensor> parts{rest_mask};
    auto ones_options = torch::TensorOptions().dtype(torch::kFloat32).device(rest_mask.device());
    const bool allow_attr_override = !(options.hybrid_hard_only && options.hybrid_override_rest_only);
    if (options.predict_opacity) {
        auto opacity_mask = torch::ones({rest_mask.size(0), 1}, ones_options);
        if (options.hybrid_hard_only)
            opacity_mask = allow_attr_override ? point_mask.clone() : torch::zeros_like(opacity_mask);
        parts.push_back(opacity_mask);
    }
    if (options.predict_scaling) {
        auto scaling_mask = torch::ones({rest_mask.size(0), 3}, ones_options);
        if (options.hybrid_hard_only)
            scaling_mask = allow_attr_override ? point_mask.expand({rest_mask.size(0), 3}) : torch::zeros_like(scaling_mask);
        parts.push_back(scaling_mask);
    }
    if (options.predict_rotation) {
        auto rotation_mask = torch::ones({rest_mask.size(0), 4}, ones_options);
        if (options.hybrid_hard_only)
            rotation_mask = allow_attr_override ? point_mask.expand({rest_mask.size(0), 4}) : torch::zeros_like(rotation_mask);
        parts.push_back(rotation_mask);
    }
    return torch::cat(parts, 1);
}

torch::Tensor buildTrainingTarget(
    const phase2_residual_field::FrozenResidualFieldPackage& frozen,
    const Phase2ResidualFieldTrainOptions& options)
{
    std::vector<torch::Tensor> parts;
    parts.push_back((frozen.features_rest_target.to(torch::kFloat32) - frozen.features_rest_base.to(torch::kFloat32)).view({frozen.xyz.size(0), -1}));
    if (options.predict_opacity)
        parts.push_back((flattenTensor(frozen.opacity.to(torch::kFloat32)) - flattenTensor(frozen.opacity_base.to(torch::kFloat32))));
    if (options.predict_scaling)
        parts.push_back((flattenTensor(frozen.scaling.to(torch::kFloat32)) - flattenTensor(frozen.scaling_base.to(torch::kFloat32))));
    if (options.predict_rotation)
        parts.push_back((flattenTensor(frozen.rotation.to(torch::kFloat32)) - flattenTensor(frozen.rotation_base.to(torch::kFloat32))));
    return torch::cat(parts, 1);
}

CompactExportOptions highPrecisionDecodedExportOptions(const Phase2ResidualFieldTrainOptions& options)
{
    CompactExportOptions export_options;
    export_options.enable_export_prune = false;
    export_options.enable_sh_bandwidth = false;
    export_options.sort_by_morton = false;
    export_options.f_rest_blockwise_quant = false;
    export_options.f_rest_locality_codec = false;
    export_options.xyz_quant_bits = options.decoded_xyz_quant_bits;
    export_options.attribute_quant_bits = options.decoded_attribute_quant_bits;
    export_options.rotation_quant_bits = options.decoded_rotation_quant_bits;
    return export_options;
}

CompactExportOptions localityBaseExportOptions(const Phase2ResidualFieldOptions& options)
{
    CompactExportOptions export_options;
    export_options.f_rest_locality_high_sh_block_size = std::max(1, options.locality_high_sh_block_size);
    export_options.f_rest_locality_low_sh_block_size = std::max(1, options.locality_low_sh_block_size);
    return export_options;
}

CompactExportOptions localityBaseExportOptions(const phase2_residual_field::FrozenResidualFieldPackage& frozen)
{
    CompactExportOptions export_options;
    export_options.f_rest_locality_high_sh_block_size = std::max(1, frozen.locality_high_sh_block_size);
    export_options.f_rest_locality_low_sh_block_size = std::max(1, frozen.locality_low_sh_block_size);
    return export_options;
}

DecodedGaussianTensors buildDecodedFromPrediction(
    const phase2_residual_field::FrozenResidualFieldPackage& frozen,
    const torch::Tensor& predicted_flat,
    const Phase2ResidualFieldTrainOptions& options)
{
    const auto slices = splitPrediction(predicted_flat, frozen.max_sh_degree, options);
    DecodedGaussianTensors decoded;
    decoded.max_sh_degree = frozen.max_sh_degree;
    decoded.active_sh_degree = frozen.active_sh_degree;
    decoded.xyz = frozen.xyz.detach().to(torch::kFloat32);
    decoded.features_dc = frozen.features_dc.detach().to(torch::kFloat32);
    auto features_rest_base = frozen.features_rest_base.defined()
        ? frozen.features_rest_base
        : torch::zeros_like(frozen.features_rest_target);
    auto predicted_features_rest = (features_rest_base + slices.rest_residual.view_as(features_rest_base)).to(torch::kFloat32);
    if (options.hybrid_hard_only && frozen.hard_point_flags.defined() && frozen.hard_point_flags.numel() == frozen.xyz.size(0)) {
        auto hard_mask = frozen.hard_point_flags.to(torch::kBool).view({frozen.xyz.size(0), 1, 1});
        decoded.features_rest = torch::where(
            hard_mask,
            predicted_features_rest,
            frozen.features_rest_target.to(torch::kFloat32)).detach();
    }
    else {
        decoded.features_rest = predicted_features_rest.detach();
    }

    if (options.predict_opacity && slices.opacity_residual.defined()) {
        auto base = frozen.opacity_base.defined() ? frozen.opacity_base : torch::zeros_like(frozen.opacity);
        auto predicted_opacity = (flattenTensor(base) + slices.opacity_residual).view_as(base).to(torch::kFloat32);
        if (options.hybrid_hard_only && frozen.hard_point_flags.defined() && frozen.hard_point_flags.numel() == frozen.xyz.size(0)) {
            if (options.hybrid_override_rest_only) {
                decoded.opacity = frozen.opacity.detach().to(torch::kFloat32);
            }
            else {
                auto hard_mask = frozen.hard_point_flags.to(torch::kBool).view({frozen.xyz.size(0), 1});
                decoded.opacity = torch::where(hard_mask, predicted_opacity, frozen.opacity.to(torch::kFloat32)).detach();
            }
        }
        else {
            decoded.opacity = predicted_opacity.detach();
        }
    }
    else {
        decoded.opacity = frozen.opacity.detach().to(torch::kFloat32);
    }

    if (options.predict_scaling && slices.scaling_residual.defined()) {
        auto base = frozen.scaling_base.defined() ? frozen.scaling_base : torch::zeros_like(frozen.scaling);
        auto predicted_scaling = (flattenTensor(base) + slices.scaling_residual).view_as(base).to(torch::kFloat32);
        if (options.hybrid_hard_only && frozen.hard_point_flags.defined() && frozen.hard_point_flags.numel() == frozen.xyz.size(0)) {
            if (options.hybrid_override_rest_only) {
                decoded.scaling = frozen.scaling.detach().to(torch::kFloat32);
            }
            else {
                auto hard_mask = frozen.hard_point_flags.to(torch::kBool).view({frozen.xyz.size(0), 1});
                decoded.scaling = torch::where(hard_mask, predicted_scaling, frozen.scaling.to(torch::kFloat32)).detach();
            }
        }
        else {
            decoded.scaling = predicted_scaling.detach();
        }
    }
    else {
        decoded.scaling = frozen.scaling.detach().to(torch::kFloat32);
    }

    if (options.predict_rotation && slices.rotation_residual.defined()) {
        auto base = frozen.rotation_base.defined() ? frozen.rotation_base : torch::zeros_like(frozen.rotation);
        auto rotation = (flattenTensor(base) + slices.rotation_residual).view_as(base).to(torch::kFloat32);
        if (options.hybrid_hard_only && frozen.hard_point_flags.defined() && frozen.hard_point_flags.numel() == frozen.xyz.size(0)) {
            if (options.hybrid_override_rest_only) {
                rotation = frozen.rotation.detach().to(torch::kFloat32);
            }
            else {
                auto hard_mask = frozen.hard_point_flags.to(torch::kBool).view({frozen.xyz.size(0), 1});
                rotation = torch::where(hard_mask, rotation, frozen.rotation.to(torch::kFloat32));
            }
        }
        decoded.rotation = maybeNormalizeQuaternionRows(rotation.view({rotation.size(0), -1})).view_as(rotation);
    }
    else {
        decoded.rotation = frozen.rotation.detach().to(torch::kFloat32);
    }
    decoded.sh_levels = frozen.sh_levels.detach().to(torch::kInt32);
    return decoded;
}

torch::Tensor fullInference(
    phase2_residual_field::Phase2ResidualField& model,
    const phase2_residual_field::FrozenResidualFieldPackage& frozen,
    const Phase2ResidualFieldTrainOptions& options,
    const torch::Tensor& prediction_mask,
    int batch_size)
{
    const auto num_points = frozen.xyz_normalized.size(0);
    const auto output_dim = model->outputDim();
    auto predictions = torch::zeros(
        {num_points, output_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(frozen.xyz_normalized.device()));

    torch::NoGradGuard no_grad;
    model->eval();
    for (int64_t start = 0; start < num_points; start += batch_size) {
        const int64_t end = std::min<int64_t>(num_points, start + batch_size);
        auto batch_pred = model->forward(
            frozen.xyz_normalized.index({torch::indexing::Slice(start, end)}),
            frozen.sh_levels.index({torch::indexing::Slice(start, end)}),
            frozen.features_dc.index({torch::indexing::Slice(start, end)}),
            phase2InputOpacity(frozen, options).index({torch::indexing::Slice(start, end)}),
            phase2InputScaling(frozen, options).index({torch::indexing::Slice(start, end)}),
            phase2InputRotation(frozen, options).index({torch::indexing::Slice(start, end)}),
            frozen.block_ids.index({torch::indexing::Slice(start, end)}));
        batch_pred = batch_pred * prediction_mask.index({torch::indexing::Slice(start, end)});
        predictions.index_put_({torch::indexing::Slice(start, end)}, batch_pred);
    }
    model->train();
    return predictions;
}

Json::Value saveFieldCheckpointForCompact(
    const phase2_residual_field::FrozenResidualFieldPackage& frozen,
    const Phase2ResidualFieldTrainOptions& options,
    const std::filesystem::path& checkpoint_path,
    const std::filesystem::path& output_path)
{
    phase2_residual_field::Phase2ResidualField model(frozen.max_sh_degree, frozen.num_blocks, options);
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint_path.string());
    model->load(archive);
    model->to(torch::Device(torch::kCPU), torch::kFloat32);
    if (options.phase2_compact_store_field_fp16)
        model->to(torch::Device(torch::kCPU), torch::kFloat16);

    torch::serialize::OutputArchive out_archive;
    model->save(out_archive);
    out_archive.save_to(output_path.string());
    const auto codec = resolveLosslessCodec(options.phase2_compact_field_zlib, options.phase2_compact_use_xz);
    return maybeCompressFileInPlace(
        output_path,
        codec,
        options.phase2_compact_field_zlib_level);
}

} // namespace

namespace phase2_residual_field
{

Phase2ResidualFieldImpl::Phase2ResidualFieldImpl(
    int max_sh_degree,
    int64_t num_blocks,
    const Phase2ResidualFieldTrainOptions& options)
    : max_sh_degree_(max_sh_degree),
      num_blocks_(num_blocks),
      num_fourier_frequencies_(options.num_fourier_frequencies),
      use_hashgrid_encoder_(options.use_hashgrid_encoder),
      include_features_dc_(options.include_features_dc),
      include_opacity_(options.include_opacity),
      include_scaling_(options.include_scaling),
      include_rotation_(options.include_rotation),
      predict_opacity_(options.predict_opacity),
      predict_scaling_(options.predict_scaling),
      predict_rotation_(options.predict_rotation),
      block_embedding_dim_(std::max(0, options.block_embedding_dim))
{
    const int rest_channels = (max_sh_degree_ + 1) * (max_sh_degree_ + 1) - 1;
    rest_output_dim_ = rest_channels * 3;
    output_dim_ = rest_output_dim_
        + (predict_opacity_ ? 1 : 0)
        + (predict_scaling_ ? 3 : 0)
        + (predict_rotation_ ? 4 : 0);

    int input_dim = 3 + (max_sh_degree_ + 1);
    if (num_fourier_frequencies_ > 0)
        input_dim += 6 * num_fourier_frequencies_;
    if (use_hashgrid_encoder_) {
        HashGridEncoderOptions hashgrid_options;
        hashgrid_options.num_levels = std::max(1, options.hashgrid_num_levels);
        hashgrid_options.features_per_level = std::max(1, options.hashgrid_features_per_level);
        hashgrid_options.log2_hashmap_size = std::max(8, options.hashgrid_log2_hashmap_size);
        hashgrid_options.base_resolution = std::max(2, options.hashgrid_base_resolution);
        hashgrid_options.per_level_scale = std::max(1.01f, options.hashgrid_per_level_scale);
        hashgrid_encoder_ = register_module("hashgrid_encoder", HashGridEncoder(hashgrid_options));
        input_dim += hashgrid_encoder_->outputDim();
    }
    if (include_features_dc_)
        input_dim += 3;
    if (include_opacity_)
        input_dim += 1;
    if (include_scaling_)
        input_dim += 3;
    if (include_rotation_)
        input_dim += 4;
    if (block_embedding_dim_ > 0 && num_blocks_ > 0) {
        block_embedding_ = register_module("block_embedding", torch::nn::Embedding(num_blocks_, block_embedding_dim_));
        input_dim += block_embedding_dim_;
    }

    const int hidden_dim = std::max(16, options.hidden_dim);
    const int num_hidden_layers = std::max(1, options.num_hidden_layers);

    network_ = register_module("network", torch::nn::Sequential());
    network_->push_back(torch::nn::Linear(input_dim, hidden_dim));
    network_->push_back(torch::nn::ReLU());
    for (int layer_idx = 1; layer_idx < num_hidden_layers; ++layer_idx) {
        network_->push_back(torch::nn::Linear(hidden_dim, hidden_dim));
        network_->push_back(torch::nn::ReLU());
    }
    network_->push_back(torch::nn::Linear(hidden_dim, output_dim_));
}

torch::Tensor Phase2ResidualFieldImpl::encode(
    const torch::Tensor& xyz_normalized,
    const torch::Tensor& sh_levels,
    const torch::Tensor& features_dc,
    const torch::Tensor& opacity,
    const torch::Tensor& scaling,
    const torch::Tensor& rotation,
    const torch::Tensor& block_ids)
{
    std::vector<torch::Tensor> features;
    features.reserve(2 * std::max(0, num_fourier_frequencies_) + 10);
    auto xyz = xyz_normalized.to(torch::kFloat32);
    features.push_back(xyz);

    if (hashgrid_encoder_)
        features.push_back(hashgrid_encoder_->forward(xyz));

    for (int frequency_idx = 0; frequency_idx < num_fourier_frequencies_; ++frequency_idx) {
        const float frequency = std::pow(2.0f, static_cast<float>(frequency_idx)) * kPi;
        features.push_back(torch::sin(xyz * frequency));
        features.push_back(torch::cos(xyz * frequency));
    }

    auto level_ids = sh_levels.to(torch::kLong).view({-1, 1});
    auto one_hot = torch::zeros(
        {xyz.size(0), max_sh_degree_ + 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(xyz.device()));
    one_hot.scatter_(1, level_ids, 1.0f);
    features.push_back(one_hot);

    if (include_features_dc_)
        features.push_back(features_dc.view({xyz.size(0), -1}).to(torch::kFloat32));
    if (include_opacity_)
        features.push_back(torch::sigmoid(opacity.view({xyz.size(0), -1}).to(torch::kFloat32)));
    if (include_scaling_)
        features.push_back(scaling.view({xyz.size(0), -1}).to(torch::kFloat32));
    if (include_rotation_)
        features.push_back(rotation.view({xyz.size(0), -1}).to(torch::kFloat32));
    if (block_embedding_ && block_embedding_dim_ > 0)
        features.push_back(block_embedding_->forward(block_ids.to(torch::kLong)));

    return torch::cat(features, 1);
}

torch::Tensor Phase2ResidualFieldImpl::forward(
    const torch::Tensor& xyz_normalized,
    const torch::Tensor& sh_levels,
    const torch::Tensor& features_dc,
    const torch::Tensor& opacity,
    const torch::Tensor& scaling,
    const torch::Tensor& rotation,
    const torch::Tensor& block_ids)
{
    return network_->forward(encode(xyz_normalized, sh_levels, features_dc, opacity, scaling, rotation, block_ids));
}

void saveFrozenPackage(
    const DecodedGaussianTensors& decoded,
    const std::filesystem::path& result_dir,
    float scene_extent,
    const Phase2ResidualFieldOptions& options)
{
    if (!decoded.xyz.defined() || decoded.xyz.numel() == 0)
        throw std::runtime_error("Cannot prepare Phase 2 package from an empty Gaussian state.");

    ensureDirectory(result_dir);

    torch::Tensor order = torch::arange(
        decoded.xyz.size(0),
        torch::TensorOptions().dtype(torch::kLong).device(decoded.xyz.device()));
    if (options.sort_by_morton)
        order = attribute_sort::mortonOrder(decoded.xyz);
    auto inverse_order = torch::empty_like(order);
    inverse_order.index_put_({order}, torch::arange(order.size(0), order.options()));

    const auto xyz_sorted = reorderLeadingDimension(decoded.xyz, order).to(torch::kFloat32);
    const auto features_dc_sorted = sanitizeTensorFinite(reorderLeadingDimension(decoded.features_dc, order));
    const auto opacity_sorted = sanitizeOpacityLogits(reorderLeadingDimension(decoded.opacity, order));
    const auto scaling_sorted = sanitizeTensorFinite(reorderLeadingDimension(decoded.scaling, order));
    const auto rotation_sorted = sanitizeTensorFinite(reorderLeadingDimension(decoded.rotation, order));
    const auto sh_levels_sorted = reorderLeadingDimension(decoded.sh_levels, order).to(torch::kInt32);
    const auto block_ids_sorted = locality_codec::computeRestBlockIds(
        sh_levels_sorted,
        decoded.max_sh_degree,
        localityBaseExportOptions(options)).to(torch::kLong);
    const int64_t num_blocks = block_ids_sorted.numel() > 0
        ? (block_ids_sorted.max().item<int64_t>() + 1)
        : 0;

    auto features_rest_target = sanitizeTensorFinite(reorderLeadingDimension(decoded.features_rest, order));
    if (options.mask_features_rest_by_sh_level)
        features_rest_target = sh_bandwidth::applyLevelsToFeaturesRest(
            features_rest_target,
            sh_levels_sorted,
            decoded.max_sh_degree);
    auto features_rest_base = torch::zeros_like(features_rest_target);
    if (options.use_locality_base) {
        features_rest_base = locality_codec::computeRestBlockBase(
            features_rest_target,
            sh_levels_sorted,
            decoded.max_sh_degree,
            localityBaseExportOptions(options)).to(torch::kFloat32);
    }
    auto opacity_base = buildPerPointBase(
        opacity_sorted.view({opacity_sorted.size(0), -1}),
        block_ids_sorted,
        num_blocks,
        false).view_as(opacity_sorted);
    auto scaling_base = buildPerPointBase(
        scaling_sorted.view({scaling_sorted.size(0), -1}),
        block_ids_sorted,
        num_blocks,
        false).view_as(scaling_sorted);
    auto rotation_base = buildPerPointBase(
        rotation_sorted.view({rotation_sorted.size(0), -1}),
        block_ids_sorted,
        num_blocks,
        true).view_as(rotation_sorted);
    const auto hybrid_selection = phase2_hybrid_selector::selectHardBlocks(
        features_rest_target,
        features_rest_base,
        sh_levels_sorted,
        block_ids_sorted,
        num_blocks,
        decoded.max_sh_degree,
        options.hybrid_selector);

    auto xyz_cpu = xyz_sorted.detach().contiguous().to(torch::kCPU);
    auto bbox_min = std::get<0>(xyz_cpu.min(0));
    auto bbox_max = std::get<0>(xyz_cpu.max(0));
    auto bbox_range = torch::clamp_min(bbox_max - bbox_min, 1e-8f);

    torch::Tensor xyz_normalized = xyz_sorted;
    if (options.normalize_xyz)
        xyz_normalized = ((xyz_sorted - bbox_min.to(xyz_sorted.device())) / bbox_range.to(xyz_sorted.device())) * 2.0f - 1.0f;

    writeTensorBinary<float>(result_dir / "xyz.bin", xyz_sorted);
    if (options.normalize_xyz)
        writeTensorBinary<c10::Half>(result_dir / "xyz_normalized.bin", xyz_normalized.to(torch::kFloat16));
    writeTensorBinary<c10::Half>(result_dir / "features_dc.bin", features_dc_sorted.to(torch::kFloat16));
    writeTensorBinary<c10::Half>(result_dir / "features_rest_base.bin", features_rest_base.to(torch::kFloat16));
    writeTensorBinary<c10::Half>(result_dir / "features_rest_target.bin", features_rest_target.to(torch::kFloat16));
    writeTensorBinary<c10::Half>(result_dir / "opacity_base.bin", opacity_base.to(torch::kFloat16));
    writeTensorBinary<c10::Half>(result_dir / "opacity.bin", opacity_sorted.to(torch::kFloat16));
    writeTensorBinary<c10::Half>(result_dir / "scaling_base.bin", scaling_base.to(torch::kFloat16));
    writeTensorBinary<c10::Half>(result_dir / "scaling.bin", scaling_sorted.to(torch::kFloat16));
    writeTensorBinary<c10::Half>(result_dir / "rotation_base.bin", rotation_base.to(torch::kFloat16));
    writeTensorBinary<c10::Half>(result_dir / "rotation.bin", rotation_sorted.to(torch::kFloat16));
    writeTensorBinary<int32_t>(result_dir / "sh_levels.bin", sh_levels_sorted.to(torch::kInt32));
    writeTensorBinary<int64_t>(result_dir / "morton_order.bin", order.to(torch::kLong));
    writeTensorBinary<int64_t>(result_dir / "inverse_morton_order.bin", inverse_order.to(torch::kLong));
    if (hybrid_selection.enabled) {
        bitpack_utils::writePackedUnsignedTensor(
            result_dir / "hard_block_flags.packed.bin",
            hybrid_selection.hard_block_flags.to(torch::kInt32),
            1u);
        bitpack_utils::writePackedUnsignedTensor(
            result_dir / "hard_point_flags.packed.bin",
            hybrid_selection.hard_point_flags.to(torch::kInt32),
            1u);
        writeTensorBinary<int64_t>(result_dir / "hard_block_ids.bin", hybrid_selection.hard_block_ids.to(torch::kLong));
        if (options.hybrid_selector.save_debug_tensors) {
            writeTensorBinary<float>(result_dir / "block_scores.bin", hybrid_selection.block_scores.to(torch::kFloat32));
            writeTensorBinary<float>(result_dir / "block_mean_mse.bin", hybrid_selection.block_mean_mse.to(torch::kFloat32));
            writeTensorBinary<float>(result_dir / "block_max_abs.bin", hybrid_selection.block_max_abs.to(torch::kFloat32));
            writeTensorBinary<int32_t>(result_dir / "block_point_counts.bin", hybrid_selection.block_point_counts.to(torch::kInt32));
            writeTensorBinary<int32_t>(result_dir / "block_levels.bin", hybrid_selection.block_levels.to(torch::kInt32));
            writeTensorBinary<float>(result_dir / "block_explicit_bytes.bin", hybrid_selection.block_explicit_bytes.to(torch::kFloat32));
            writeTensorBinary<float>(result_dir / "block_explicit_bpp.bin", hybrid_selection.block_explicit_bpp.to(torch::kFloat32));
        }
        writeMetadata(result_dir / "hybrid_selector_summary.json", hybridSelectorSummaryJson(hybrid_selection));
    }

    Json::Value root;
    root["format"] = "phase2_residual_field_frozen";
    root["topology_frozen"] = true;
    root["scene_extent"] = scene_extent;
    root["num_points"] = Json::Value::Int64(xyz_sorted.size(0));
    root["max_sh_degree"] = decoded.max_sh_degree;
    root["active_sh_degree"] = decoded.active_sh_degree;
    root["sorted_by_morton"] = options.sort_by_morton;
    root["normalized_xyz"] = options.normalize_xyz;
    root["mask_features_rest_by_sh_level"] = options.mask_features_rest_by_sh_level;
    root["use_locality_base"] = options.use_locality_base;
    root["locality_high_sh_block_size"] = options.locality_high_sh_block_size;
    root["locality_low_sh_block_size"] = options.locality_low_sh_block_size;
    root["num_blocks"] = Json::Value::Int64(num_blocks);
    root["xyz_shape"] = tensorShapeJson(xyz_sorted);
    root["xyz_normalized_shape"] = options.normalize_xyz ? tensorShapeJson(xyz_normalized) : Json::Value(Json::nullValue);
    root["features_dc_shape"] = tensorShapeJson(features_dc_sorted);
    root["features_rest_base_shape"] = tensorShapeJson(features_rest_base);
    root["features_rest_target_shape"] = tensorShapeJson(features_rest_target);
    root["opacity_base_shape"] = tensorShapeJson(opacity_base);
    root["opacity_shape"] = tensorShapeJson(opacity_sorted);
    root["scaling_base_shape"] = tensorShapeJson(scaling_base);
    root["scaling_shape"] = tensorShapeJson(scaling_sorted);
    root["rotation_base_shape"] = tensorShapeJson(rotation_base);
    root["rotation_shape"] = tensorShapeJson(rotation_sorted);
    root["sh_levels_shape"] = tensorShapeJson(sh_levels_sorted);
    root["block_ids_shape"] = Json::Value(Json::nullValue);
    root["morton_order_shape"] = tensorShapeJson(order);
    root["inverse_morton_order_shape"] = tensorShapeJson(inverse_order);
    Json::Value hybrid_selector_json;
    hybrid_selector_json["enabled"] = options.hybrid_selector.enable;
    hybrid_selector_json["hard_point_ratio"] = options.hybrid_selector.hard_point_ratio;
    hybrid_selector_json["alpha"] = options.hybrid_selector.alpha;
    hybrid_selector_json["beta"] = options.hybrid_selector.beta;
    hybrid_selector_json["gamma"] = options.hybrid_selector.gamma;
    hybrid_selector_json["delta"] = options.hybrid_selector.delta;
    hybrid_selector_json["explicit_cost_int4_rel_mse_threshold"] =
        options.hybrid_selector.explicit_cost_int4_rel_mse_threshold;
    hybrid_selector_json["min_hard_blocks"] = options.hybrid_selector.min_hard_blocks;
    hybrid_selector_json["max_hard_blocks"] = options.hybrid_selector.max_hard_blocks;
    hybrid_selector_json["save_debug_tensors"] = options.hybrid_selector.save_debug_tensors;
    if (hybrid_selection.enabled) {
        hybrid_selector_json["hard_block_flags_storage"] = "packed_uint";
        hybrid_selector_json["hard_block_flags_bits"] = 1;
        hybrid_selector_json["hard_block_flags_shape"] = tensorShapeJson(hybrid_selection.hard_block_flags);
        hybrid_selector_json["hard_point_flags_storage"] = "packed_uint";
        hybrid_selector_json["hard_point_flags_bits"] = 1;
        hybrid_selector_json["hard_point_flags_shape"] = tensorShapeJson(hybrid_selection.hard_point_flags);
        hybrid_selector_json["hard_block_ids_shape"] = tensorShapeJson(hybrid_selection.hard_block_ids);
        hybrid_selector_json["num_hard_blocks"] = Json::Value::Int64(hybrid_selection.num_hard_blocks);
        hybrid_selector_json["num_hard_points"] = Json::Value::Int64(hybrid_selection.num_hard_points);
        hybrid_selector_json["realized_hard_point_ratio"] = hybrid_selection.realized_hard_point_ratio;
        hybrid_selector_json["realized_hard_block_ratio"] = hybrid_selection.realized_hard_block_ratio;
        hybrid_selector_json["summary"] = hybridSelectorSummaryJson(hybrid_selection);
        if (options.hybrid_selector.save_debug_tensors) {
            hybrid_selector_json["block_scores_shape"] = tensorShapeJson(hybrid_selection.block_scores);
            hybrid_selector_json["block_mean_mse_shape"] = tensorShapeJson(hybrid_selection.block_mean_mse);
            hybrid_selector_json["block_max_abs_shape"] = tensorShapeJson(hybrid_selection.block_max_abs);
            hybrid_selector_json["block_point_counts_shape"] = tensorShapeJson(hybrid_selection.block_point_counts);
            hybrid_selector_json["block_levels_shape"] = tensorShapeJson(hybrid_selection.block_levels);
            hybrid_selector_json["block_explicit_bytes_shape"] = tensorShapeJson(hybrid_selection.block_explicit_bytes);
            hybrid_selector_json["block_explicit_bpp_shape"] = tensorShapeJson(hybrid_selection.block_explicit_bpp);
        }
    }
    root["hybrid_selector"] = hybrid_selector_json;

    Json::Value bbox_min_json(Json::arrayValue);
    Json::Value bbox_max_json(Json::arrayValue);
    auto bbox_min_cpu = bbox_min.contiguous();
    auto bbox_max_cpu = bbox_max.contiguous();
    const float* bbox_min_ptr = bbox_min_cpu.data_ptr<float>();
    const float* bbox_max_ptr = bbox_max_cpu.data_ptr<float>();
    for (int axis = 0; axis < 3; ++axis) {
        bbox_min_json.append(bbox_min_ptr[axis]);
        bbox_max_json.append(bbox_max_ptr[axis]);
    }
    root["xyz_bbox_min"] = bbox_min_json;
    root["xyz_bbox_max"] = bbox_max_json;

    writeMetadata(result_dir / "metadata.json", root);
}

FrozenResidualFieldPackage loadFrozenPackage(
    const std::filesystem::path& input_path,
    torch::DeviceType device_type)
{
    const auto result_dir = std::filesystem::is_directory(input_path) ? input_path : input_path.parent_path();
    const auto metadata_path = std::filesystem::is_directory(input_path) ? (input_path / "metadata.json") : input_path;
    const auto meta = readMetadataJson(metadata_path);

    FrozenResidualFieldPackage frozen;
    frozen.max_sh_degree = meta["max_sh_degree"].asInt();
    frozen.active_sh_degree = meta["active_sh_degree"].asInt();
    frozen.scene_extent = meta["scene_extent"].asFloat();
    frozen.use_locality_base = meta.get("use_locality_base", false).asBool();
    frozen.locality_high_sh_block_size = meta.get("locality_high_sh_block_size", 64).asInt();
    frozen.locality_low_sh_block_size = meta.get("locality_low_sh_block_size", 128).asInt();
    frozen.num_blocks = meta.get("num_blocks", 0).asInt64();

    frozen.xyz = readTensorBinary<float>(
        result_dir / "xyz.bin",
        tensorShapeFromJson(meta["xyz_shape"]),
        torch::kFloat32,
        device_type);
    if (meta["normalized_xyz"].asBool() && std::filesystem::exists(result_dir / "xyz_normalized.bin")) {
        frozen.xyz_normalized = readTensorBinary<c10::Half>(
            result_dir / "xyz_normalized.bin",
            tensorShapeFromJson(meta["xyz_normalized_shape"]),
            torch::kFloat16,
            device_type).to(torch::kFloat32);
    }
    else {
        frozen.xyz_normalized = normalizeFromBounds(frozen.xyz, meta["xyz_bbox_min"], meta["xyz_bbox_max"]);
    }
    frozen.features_dc = readTensorBinary<c10::Half>(
        result_dir / "features_dc.bin",
        tensorShapeFromJson(meta["features_dc_shape"]),
        torch::kFloat16,
        device_type).to(torch::kFloat32);
    frozen.features_dc = sanitizeTensorFinite(frozen.features_dc);
    if (std::filesystem::exists(result_dir / "features_rest_base.bin")) {
        frozen.features_rest_base = readTensorBinary<c10::Half>(
            result_dir / "features_rest_base.bin",
            tensorShapeFromJson(meta["features_rest_base_shape"]),
            torch::kFloat16,
            device_type).to(torch::kFloat32);
    }
    frozen.features_rest_target = readTensorBinary<c10::Half>(
        result_dir / "features_rest_target.bin",
        tensorShapeFromJson(meta["features_rest_target_shape"]),
        torch::kFloat16,
        device_type).to(torch::kFloat32);
    frozen.features_rest_target = sanitizeTensorFinite(frozen.features_rest_target);
    if (!frozen.features_rest_base.defined())
        frozen.features_rest_base = torch::zeros_like(frozen.features_rest_target);
    else
        frozen.features_rest_base = sanitizeTensorFinite(frozen.features_rest_base);
    frozen.sh_levels = readTensorBinary<int32_t>(
        result_dir / "sh_levels.bin",
        tensorShapeFromJson(meta["sh_levels_shape"]),
        torch::kInt32,
        device_type);
    frozen.block_ids = locality_codec::computeRestBlockIds(
        frozen.sh_levels,
        frozen.max_sh_degree,
        localityBaseExportOptions(frozen));
    if (frozen.num_blocks <= 0 && frozen.block_ids.defined() && frozen.block_ids.numel() > 0)
        frozen.num_blocks = frozen.block_ids.max().item<int64_t>() + 1;
    if (std::filesystem::exists(result_dir / "opacity_base.bin")) {
        frozen.opacity_base = readTensorBinary<c10::Half>(
            result_dir / "opacity_base.bin",
            tensorShapeFromJson(meta["opacity_base_shape"]),
            torch::kFloat16,
            device_type).to(torch::kFloat32);
    }
    frozen.opacity = readTensorBinary<c10::Half>(
        result_dir / "opacity.bin",
        tensorShapeFromJson(meta["opacity_shape"]),
        torch::kFloat16,
        device_type).to(torch::kFloat32);
    frozen.opacity = sanitizeOpacityLogits(frozen.opacity);
    if (!frozen.opacity_base.defined())
        frozen.opacity_base = buildPerPointBase(
            flattenTensor(frozen.opacity),
            frozen.block_ids,
            frozen.num_blocks,
            false).view_as(frozen.opacity);
    else
        frozen.opacity_base = sanitizeOpacityLogits(frozen.opacity_base);
    if (std::filesystem::exists(result_dir / "scaling_base.bin")) {
        frozen.scaling_base = readTensorBinary<c10::Half>(
            result_dir / "scaling_base.bin",
            tensorShapeFromJson(meta["scaling_base_shape"]),
            torch::kFloat16,
            device_type).to(torch::kFloat32);
    }
    frozen.scaling = readTensorBinary<c10::Half>(
        result_dir / "scaling.bin",
        tensorShapeFromJson(meta["scaling_shape"]),
        torch::kFloat16,
        device_type).to(torch::kFloat32);
    frozen.scaling = sanitizeTensorFinite(frozen.scaling);
    if (!frozen.scaling_base.defined())
        frozen.scaling_base = buildPerPointBase(
            flattenTensor(frozen.scaling),
            frozen.block_ids,
            frozen.num_blocks,
            false).view_as(frozen.scaling);
    else
        frozen.scaling_base = sanitizeTensorFinite(frozen.scaling_base);
    if (std::filesystem::exists(result_dir / "rotation_base.bin")) {
        frozen.rotation_base = readTensorBinary<c10::Half>(
            result_dir / "rotation_base.bin",
            tensorShapeFromJson(meta["rotation_base_shape"]),
            torch::kFloat16,
            device_type).to(torch::kFloat32);
    }
    frozen.rotation = readTensorBinary<c10::Half>(
        result_dir / "rotation.bin",
        tensorShapeFromJson(meta["rotation_shape"]),
        torch::kFloat16,
        device_type).to(torch::kFloat32);
    frozen.rotation = sanitizeTensorFinite(frozen.rotation);
    if (!frozen.rotation_base.defined())
        frozen.rotation_base = buildPerPointBase(
            flattenTensor(frozen.rotation),
            frozen.block_ids,
            frozen.num_blocks,
            true).view_as(frozen.rotation);
    else
        frozen.rotation_base = maybeNormalizeQuaternionRows(flattenTensor(sanitizeTensorFinite(frozen.rotation_base))).view_as(frozen.rotation_base);
    frozen.morton_order = readTensorBinary<int64_t>(
        result_dir / "morton_order.bin",
        tensorShapeFromJson(meta["morton_order_shape"]),
        torch::kLong,
        device_type);
    frozen.inverse_morton_order = readTensorBinary<int64_t>(
        result_dir / "inverse_morton_order.bin",
        tensorShapeFromJson(meta["inverse_morton_order_shape"]),
        torch::kLong,
        device_type);
    const auto hybrid_selector_meta = meta["hybrid_selector"];
    if (!hybrid_selector_meta.isNull() && hybrid_selector_meta.get("enabled", false).asBool()) {
        if (std::filesystem::exists(result_dir / "hard_block_flags.packed.bin")) {
            frozen.hard_block_flags = bitpack_utils::readPackedUnsignedTensor(
                result_dir / "hard_block_flags.packed.bin",
                elementCount(tensorShapeFromJson(hybrid_selector_meta["hard_block_flags_shape"])),
                static_cast<std::uint8_t>(hybrid_selector_meta.get("hard_block_flags_bits", 1).asUInt()),
                device_type).to(torch::kInt32);
        }
        if (std::filesystem::exists(result_dir / "hard_point_flags.packed.bin")) {
            frozen.hard_point_flags = bitpack_utils::readPackedUnsignedTensor(
                result_dir / "hard_point_flags.packed.bin",
                elementCount(tensorShapeFromJson(hybrid_selector_meta["hard_point_flags_shape"])),
                static_cast<std::uint8_t>(hybrid_selector_meta.get("hard_point_flags_bits", 1).asUInt()),
                device_type).to(torch::kInt32);
        }
        if (std::filesystem::exists(result_dir / "hard_block_ids.bin")) {
            frozen.hard_block_ids = readTensorBinary<int64_t>(
                result_dir / "hard_block_ids.bin",
                tensorShapeFromJson(hybrid_selector_meta["hard_block_ids_shape"]),
                torch::kLong,
                device_type);
        }
        if (std::filesystem::exists(result_dir / "block_scores.bin")) {
            frozen.block_scores = readTensorBinary<float>(
                result_dir / "block_scores.bin",
                tensorShapeFromJson(hybrid_selector_meta["block_scores_shape"]),
                torch::kFloat32,
                device_type);
            frozen.block_mean_mse = readTensorBinary<float>(
                result_dir / "block_mean_mse.bin",
                tensorShapeFromJson(hybrid_selector_meta["block_mean_mse_shape"]),
                torch::kFloat32,
                device_type);
            frozen.block_max_abs = readTensorBinary<float>(
                result_dir / "block_max_abs.bin",
                tensorShapeFromJson(hybrid_selector_meta["block_max_abs_shape"]),
                torch::kFloat32,
                device_type);
            frozen.block_point_counts = readTensorBinary<int32_t>(
                result_dir / "block_point_counts.bin",
                tensorShapeFromJson(hybrid_selector_meta["block_point_counts_shape"]),
                torch::kInt32,
                device_type);
            frozen.block_levels = readTensorBinary<int32_t>(
                result_dir / "block_levels.bin",
                tensorShapeFromJson(hybrid_selector_meta["block_levels_shape"]),
                torch::kInt32,
                device_type);
        }
        frozen.num_hard_blocks = hybrid_selector_meta.get("num_hard_blocks", 0).asInt64();
        frozen.num_hard_points = hybrid_selector_meta.get("num_hard_points", 0).asInt64();
    }

    return frozen;
}

void savePhase2CompactPackage(
    const FrozenResidualFieldPackage& frozen,
    const Phase2ResidualFieldTrainOptions& options,
    const std::filesystem::path& checkpoint_path,
    const std::filesystem::path& result_dir)
{
    if (!std::filesystem::exists(checkpoint_path))
        throw std::runtime_error("Cannot find Phase 2 checkpoint at " + checkpoint_path.string());

    ensureDirectory(result_dir);
    const auto geometry_codec_name = resolveLosslessCodec(options.phase2_compact_geometry_zlib, options.phase2_compact_use_xz);
    const auto quantized_tensor_codec = resolveLosslessCodec(options.phase2_compact_quantized_tensor_zlib, options.phase2_compact_use_xz);
    const auto rest_block_bases = blockMeansFromPerPointBase(
        frozen.features_rest_base.to(torch::kFloat32),
        frozen.block_ids,
        frozen.num_blocks,
        false).to(torch::kFloat16);
    const auto opacity_block_bases = blockMeansFromPerPointBase(
        frozen.opacity_base.to(torch::kFloat32),
        frozen.block_ids,
        frozen.num_blocks,
        false).to(torch::kFloat16);
    const auto scaling_block_bases = blockMeansFromPerPointBase(
        frozen.scaling_base.to(torch::kFloat32),
        frozen.block_ids,
        frozen.num_blocks,
        false).to(torch::kFloat16);
    const auto rotation_block_bases = blockMeansFromPerPointBase(
        frozen.rotation_base.to(torch::kFloat32),
        frozen.block_ids,
        frozen.num_blocks,
        true).to(torch::kFloat16);

    Json::Value xyz_meta;
    if (options.phase2_compact_use_geometry_codec) {
        geometry_codec::GeometryCodecOptions codec_options;
        codec_options.quant_bits = options.phase2_compact_geometry_quant_bits;
        geometry_codec::encodeMortonDelta(frozen.xyz, result_dir / "xyz.geom", xyz_meta, codec_options);
        xyz_meta["file_storage"] = maybeCompressFileInPlace(
            result_dir / "xyz.geom",
            geometry_codec_name,
            options.phase2_compact_geometry_zlib_level);
    }
    else {
        writeTensorBinary<c10::Half>(result_dir / "xyz.bin", frozen.xyz.to(torch::kFloat16));
        xyz_meta["storage"] = "fp16";
        xyz_meta["shape"] = tensorShapeJson(frozen.xyz);
        Json::Value bbox_min_json(Json::arrayValue);
        Json::Value bbox_max_json(Json::arrayValue);
        auto bbox_min = std::get<0>(frozen.xyz.detach().to(torch::kCPU, torch::kFloat32).min(0)).contiguous();
        auto bbox_max = std::get<0>(frozen.xyz.detach().to(torch::kCPU, torch::kFloat32).max(0)).contiguous();
        const float* bbox_min_ptr = bbox_min.data_ptr<float>();
        const float* bbox_max_ptr = bbox_max.data_ptr<float>();
        for (int axis = 0; axis < 3; ++axis) {
            bbox_min_json.append(bbox_min_ptr[axis]);
            bbox_max_json.append(bbox_max_ptr[axis]);
        }
        xyz_meta["bbox_min"] = bbox_min_json;
        xyz_meta["bbox_max"] = bbox_max_json;
    }

    Json::Value fdc_meta;
    saveQuantizedTensorUint(
        result_dir / "features_dc.bin",
        frozen.features_dc.to(torch::kFloat32),
        options.phase2_compact_fdc_quant_bits,
        fdc_meta,
        quantized_tensor_codec,
        options.phase2_compact_quantized_tensor_zlib_level);

    Json::Value sh_levels_meta;
    const auto max_level_value = static_cast<std::uint32_t>(std::max(0, frozen.max_sh_degree));
    const auto sh_bits = options.phase2_compact_pack_sh_levels
        ? bitpack_utils::minimumBitsForValue(max_level_value)
        : 32u;
    if (options.phase2_compact_pack_sh_levels) {
        bitpack_utils::writePackedUnsignedTensor(
            result_dir / "sh_levels.packed.bin",
            frozen.sh_levels.to(torch::kInt32),
            sh_bits);
        sh_levels_meta["storage"] = "packed_uint";
        sh_levels_meta["bits"] = sh_bits;
    }
    else {
        writeTensorBinary<int32_t>(result_dir / "sh_levels.bin", frozen.sh_levels.to(torch::kInt32));
        sh_levels_meta["storage"] = "int32";
    }
    sh_levels_meta["shape"] = tensorShapeJson(frozen.sh_levels);
    sh_levels_meta["max_value"] = static_cast<int>(max_level_value);

    writeTensorBinary<c10::Half>(result_dir / "features_rest_block_bases.bin", rest_block_bases);
    if (options.predict_opacity)
        writeTensorBinary<c10::Half>(result_dir / "opacity_block_bases.bin", opacity_block_bases);
    else
        writeTensorBinary<c10::Half>(result_dir / "opacity.bin", frozen.opacity.to(torch::kFloat16));
    if (options.predict_scaling)
        writeTensorBinary<c10::Half>(result_dir / "scaling_block_bases.bin", scaling_block_bases);
    else
        writeTensorBinary<c10::Half>(result_dir / "scaling.bin", frozen.scaling.to(torch::kFloat16));
    if (options.predict_rotation)
        writeTensorBinary<c10::Half>(result_dir / "rotation_block_bases.bin", rotation_block_bases);
    else
        writeTensorBinary<c10::Half>(result_dir / "rotation.bin", frozen.rotation.to(torch::kFloat16));

    Json::Value field_checkpoint_meta = saveFieldCheckpointForCompact(
        frozen,
        options,
        checkpoint_path,
        result_dir / "field_weights.pt");

    Json::Value phase2_field;
    phase2_field["num_fourier_frequencies"] = options.num_fourier_frequencies;
    phase2_field["use_hashgrid_encoder"] = options.use_hashgrid_encoder;
    phase2_field["hashgrid_num_levels"] = options.hashgrid_num_levels;
    phase2_field["hashgrid_features_per_level"] = options.hashgrid_features_per_level;
    phase2_field["hashgrid_log2_hashmap_size"] = options.hashgrid_log2_hashmap_size;
    phase2_field["hashgrid_base_resolution"] = options.hashgrid_base_resolution;
    phase2_field["hashgrid_per_level_scale"] = options.hashgrid_per_level_scale;
    phase2_field["hidden_dim"] = options.hidden_dim;
    phase2_field["num_hidden_layers"] = options.num_hidden_layers;
    phase2_field["batch_size"] = options.batch_size;
    phase2_field["max_steps"] = options.max_steps;
    phase2_field["log_interval"] = options.log_interval;
    phase2_field["eval_interval"] = options.eval_interval;
    phase2_field["learning_rate"] = options.learning_rate;
    phase2_field["weight_decay"] = options.weight_decay;
    phase2_field["include_features_dc"] = options.include_features_dc;
    phase2_field["include_opacity"] = options.include_opacity;
    phase2_field["include_scaling"] = options.include_scaling;
    phase2_field["include_rotation"] = options.include_rotation;
    phase2_field["predict_opacity"] = options.predict_opacity;
    phase2_field["predict_scaling"] = options.predict_scaling;
    phase2_field["predict_rotation"] = options.predict_rotation;
    phase2_field["block_embedding_dim"] = options.block_embedding_dim;
    phase2_field["hybrid_hard_only"] = options.hybrid_hard_only;
    phase2_field["hybrid_override_rest_only"] = options.hybrid_override_rest_only;
    phase2_field["hybrid_easy_export_sh_drop"] = options.hybrid_easy_export_sh_drop;
    phase2_field["hybrid_easy_export_sh_preserve_blocks"] = options.hybrid_easy_export_sh_preserve_blocks;
    phase2_field["hybrid_easy_export_sh_energy_keep_ratio"] = options.hybrid_easy_export_sh_energy_keep_ratio;
    phase2_field["hybrid_easy_export_sh_min_opacity"] = options.hybrid_easy_export_sh_min_opacity;
    phase2_field["hybrid_easy_export_sh_min_level"] = options.hybrid_easy_export_sh_min_level;
    phase2_field["save_decoded_compact"] = options.save_decoded_compact;
    phase2_field["save_phase2_compact"] = options.save_phase2_compact;
    phase2_field["decoded_xyz_quant_bits"] = options.decoded_xyz_quant_bits;
    phase2_field["decoded_attribute_quant_bits"] = options.decoded_attribute_quant_bits;
    phase2_field["decoded_rotation_quant_bits"] = options.decoded_rotation_quant_bits;
    phase2_field["phase2_compact_opacity_quant_bits"] = options.phase2_compact_opacity_quant_bits;
    phase2_field["phase2_compact_scaling_quant_bits"] = options.phase2_compact_scaling_quant_bits;
    phase2_field["phase2_compact_pack_sh_levels"] = options.phase2_compact_pack_sh_levels;
    phase2_field["phase2_compact_fdc_quant_bits"] = options.phase2_compact_fdc_quant_bits;
    phase2_field["phase2_compact_easy_rest_base_quant_bits"] = options.phase2_compact_easy_rest_base_quant_bits;
    phase2_field["phase2_compact_easy_rest_scale_quant_bits"] = options.phase2_compact_easy_rest_scale_quant_bits;
    phase2_field["phase2_compact_easy_rest_int2_rel_mse_threshold"] = options.phase2_compact_easy_rest_int2_rel_mse_threshold;
    phase2_field["phase2_compact_use_geometry_codec"] = options.phase2_compact_use_geometry_codec;
    phase2_field["phase2_compact_geometry_quant_bits"] = options.phase2_compact_geometry_quant_bits;
    phase2_field["phase2_compact_store_field_fp16"] = options.phase2_compact_store_field_fp16;
    phase2_field["phase2_compact_quantized_tensor_zlib"] = options.phase2_compact_quantized_tensor_zlib;
    phase2_field["phase2_compact_quantized_tensor_zlib_level"] = options.phase2_compact_quantized_tensor_zlib_level;
    phase2_field["phase2_compact_geometry_zlib"] = options.phase2_compact_geometry_zlib;
    phase2_field["phase2_compact_geometry_zlib_level"] = options.phase2_compact_geometry_zlib_level;
    phase2_field["phase2_compact_field_zlib"] = options.phase2_compact_field_zlib;
    phase2_field["phase2_compact_field_zlib_level"] = options.phase2_compact_field_zlib_level;
    phase2_field["phase2_compact_use_xz"] = options.phase2_compact_use_xz;

    Json::Value root;
    root["format"] = "phase2_residual_field_compact";
    root["representation"] = "light_anchors_plus_field_weights_v3";
    root["field_checkpoint"] = "field_weights.pt";
    root["field_checkpoint_storage"] = field_checkpoint_meta;
    root["scene_extent"] = frozen.scene_extent;
    root["num_points"] = Json::Value::Int64(frozen.xyz.size(0));
    root["max_sh_degree"] = frozen.max_sh_degree;
    root["active_sh_degree"] = frozen.active_sh_degree;
    root["use_locality_base"] = frozen.use_locality_base;
    root["locality_high_sh_block_size"] = frozen.locality_high_sh_block_size;
    root["locality_low_sh_block_size"] = frozen.locality_low_sh_block_size;
    root["num_blocks"] = Json::Value::Int64(frozen.num_blocks);
    root["xyz"] = xyz_meta;
    root["features_dc"] = fdc_meta;
    root["sh_levels"] = sh_levels_meta;
    root["features_rest_block_bases_shape"] = tensorShapeJson(rest_block_bases);
    root["opacity_storage"] = options.predict_opacity ? "field_predicted" : "fp16";
    root["scaling_storage"] = options.predict_scaling ? "field_predicted" : "fp16";
    root["rotation_storage"] = options.predict_rotation ? "field_predicted" : "fp16";
    if (options.predict_opacity)
        root["opacity_block_bases_shape"] = tensorShapeJson(opacity_block_bases);
    else
        root["opacity_shape"] = tensorShapeJson(frozen.opacity);
    if (options.predict_scaling)
        root["scaling_block_bases_shape"] = tensorShapeJson(scaling_block_bases);
    else
        root["scaling_shape"] = tensorShapeJson(frozen.scaling);
    if (options.predict_rotation)
        root["rotation_block_bases_shape"] = tensorShapeJson(rotation_block_bases);
    else
        root["rotation_shape"] = tensorShapeJson(frozen.rotation);
    root["phase2_field"] = phase2_field;

    writeMetadata(result_dir / "metadata.json", root);
}

void saveSelectiveHybridPackage(
    const FrozenResidualFieldPackage& frozen,
    const DecodedGaussianTensors& decoded,
    const Phase2ResidualFieldTrainOptions& options,
    const std::filesystem::path& checkpoint_path,
    const std::filesystem::path& result_dir)
{
    if (!std::filesystem::exists(checkpoint_path))
        throw std::runtime_error("Cannot find Phase 2 checkpoint at " + checkpoint_path.string());
    if (!frozen.hard_point_flags.defined() || frozen.hard_point_flags.numel() != frozen.xyz.size(0))
        throw std::runtime_error("Selective hybrid export expects valid hard_point_flags.");
    if (!frozen.hard_block_ids.defined())
        throw std::runtime_error("Selective hybrid export expects valid hard_block_ids.");

    ensureDirectory(result_dir);
    const auto geometry_codec_name = resolveLosslessCodec(options.phase2_compact_geometry_zlib, options.phase2_compact_use_xz);
    const auto quantized_tensor_codec = resolveLosslessCodec(options.phase2_compact_quantized_tensor_zlib, options.phase2_compact_use_xz);
    const auto easy_rest_codec = resolveLosslessCodec(options.phase2_compact_easy_rest_zlib, options.phase2_compact_use_xz);

    Json::Value xyz_meta;
    if (options.phase2_compact_use_geometry_codec) {
        geometry_codec::GeometryCodecOptions codec_options;
        codec_options.quant_bits = options.phase2_compact_geometry_quant_bits;
        geometry_codec::encodeMortonDelta(frozen.xyz, result_dir / "xyz.geom", xyz_meta, codec_options);
        xyz_meta["file_storage"] = maybeCompressFileInPlace(
            result_dir / "xyz.geom",
            geometry_codec_name,
            options.phase2_compact_geometry_zlib_level);
    }
    else {
        writeTensorBinary<c10::Half>(result_dir / "xyz.bin", frozen.xyz.to(torch::kFloat16));
        xyz_meta["storage"] = "fp16";
        xyz_meta["shape"] = tensorShapeJson(frozen.xyz);
        Json::Value bbox_min_json(Json::arrayValue);
        Json::Value bbox_max_json(Json::arrayValue);
        auto bbox_min = std::get<0>(frozen.xyz.detach().to(torch::kCPU, torch::kFloat32).min(0)).contiguous();
        auto bbox_max = std::get<0>(frozen.xyz.detach().to(torch::kCPU, torch::kFloat32).max(0)).contiguous();
        const float* bbox_min_ptr = bbox_min.data_ptr<float>();
        const float* bbox_max_ptr = bbox_max.data_ptr<float>();
        for (int axis = 0; axis < 3; ++axis) {
            bbox_min_json.append(bbox_min_ptr[axis]);
            bbox_max_json.append(bbox_max_ptr[axis]);
        }
        xyz_meta["bbox_min"] = bbox_min_json;
        xyz_meta["bbox_max"] = bbox_max_json;
    }

    Json::Value fdc_meta;
    saveQuantizedTensorUint(
        result_dir / "features_dc.bin",
        frozen.features_dc.to(torch::kFloat32),
        options.phase2_compact_fdc_quant_bits,
        fdc_meta,
        quantized_tensor_codec,
        options.phase2_compact_quantized_tensor_zlib_level);

    Json::Value sh_levels_meta;
    const auto max_level_value = static_cast<std::uint32_t>(std::max(0, frozen.max_sh_degree));
    const auto sh_bits = options.phase2_compact_pack_sh_levels
        ? bitpack_utils::minimumBitsForValue(max_level_value)
        : 32u;
    if (options.phase2_compact_pack_sh_levels) {
        bitpack_utils::writePackedUnsignedTensor(
            result_dir / "sh_levels.packed.bin",
            frozen.sh_levels.to(torch::kInt32),
            sh_bits);
        sh_levels_meta["storage"] = "packed_uint";
        sh_levels_meta["bits"] = sh_bits;
    }
    else {
        writeTensorBinary<int32_t>(result_dir / "sh_levels.bin", frozen.sh_levels.to(torch::kInt32));
        sh_levels_meta["storage"] = "int32";
    }
    sh_levels_meta["shape"] = tensorShapeJson(frozen.sh_levels);
    sh_levels_meta["max_value"] = static_cast<int>(max_level_value);

    Json::Value opacity_meta;
    saveQuantizedTensorUint(
        result_dir / "opacity.bin",
        torch::clamp(torch::sigmoid(frozen.opacity.to(torch::kFloat32)), 1e-6f, 1.0f - 1e-6f),
        std::clamp(options.phase2_compact_opacity_quant_bits, 1, 8),
        opacity_meta,
        quantized_tensor_codec,
        options.phase2_compact_quantized_tensor_zlib_level);
    opacity_meta["representation"] = "activation";

    Json::Value scaling_meta;
    saveQuantizedTensorUint(
        result_dir / "scaling.bin",
        frozen.scaling.to(torch::kFloat32),
        std::clamp(options.phase2_compact_scaling_quant_bits, 1, 8),
        scaling_meta,
        quantized_tensor_codec,
        options.phase2_compact_quantized_tensor_zlib_level);

    Json::Value rotation_meta;
    saveQuantizedTensorUint(
        result_dir / "rotation.bin",
        frozen.rotation.to(torch::kFloat32),
        std::min(options.decoded_rotation_quant_bits, 8),
        rotation_meta,
        quantized_tensor_codec,
        options.phase2_compact_quantized_tensor_zlib_level);

    bitpack_utils::writePackedUnsignedTensor(
        result_dir / "hard_point_flags.packed.bin",
        frozen.hard_point_flags.to(torch::kInt32),
        1u);
    writeTensorBinary<int64_t>(
        result_dir / "hard_block_ids.bin",
        frozen.hard_block_ids.to(torch::kLong));

    const auto hard_indices = indicesFromPackedFlags(frozen.hard_point_flags, true);
    const auto easy_indices = indicesFromPackedFlags(frozen.hard_point_flags, false);
    const auto easy_original_block_ids = frozen.block_ids.index_select(0, easy_indices).to(torch::kLong);
    const auto remapped_easy = remapBlockIdsContiguous(easy_original_block_ids);
    const auto easy_block_ids = remapped_easy.first;
    const int64_t num_easy_blocks = remapped_easy.second;
    const auto easy_sh_levels = frozen.sh_levels.index_select(0, easy_indices);
    auto easy_export_sh_levels = easy_sh_levels.clone();
    const auto easy_features_rest = decoded.features_rest.index_select(0, easy_indices);
    auto easy_features_rest_export = easy_features_rest.clone();
    if (options.hybrid_easy_export_sh_drop && easy_indices.numel() > 0) {
        const auto easy_opacity = frozen.opacity.index_select(0, easy_indices);
        easy_export_sh_levels = estimateBlockwiseExportLevels(
            easy_features_rest,
            easy_opacity,
            easy_block_ids,
            num_easy_blocks,
            frozen.max_sh_degree,
            options.hybrid_easy_export_sh_energy_keep_ratio,
            options.hybrid_easy_export_sh_min_opacity,
            std::clamp(options.hybrid_easy_export_sh_min_level, 0, frozen.max_sh_degree));
        easy_features_rest_export = sh_bandwidth::applyLevelsToFeaturesRest(
            easy_features_rest,
            easy_export_sh_levels,
            frozen.max_sh_degree);
    }
    auto easy_export_options = localityBaseExportOptions(frozen);
    easy_export_options.f_rest_locality_codec = true;
    easy_export_options.f_rest_locality_int2_rel_mse_threshold = options.phase2_compact_easy_rest_int2_rel_mse_threshold;
    const auto easy_encoded = (options.hybrid_easy_export_sh_drop && options.hybrid_easy_export_sh_preserve_blocks)
        ? locality_codec::encodeRestPayloadPreserveBlocks(
            easy_features_rest_export,
            easy_block_ids,
            easy_export_sh_levels,
            frozen.max_sh_degree,
            easy_export_options)
        : locality_codec::encodeRestPayload(
            easy_features_rest_export,
            easy_export_sh_levels,
            frozen.max_sh_degree,
            easy_export_options);
    const auto easy_rest_meta = saveEncodedRestPayload(
        result_dir,
        "easy_rest",
        easy_encoded,
        easy_rest_codec,
        options.phase2_compact_easy_rest_zlib_level,
        options.phase2_compact_easy_rest_base_quant_bits,
        options.phase2_compact_easy_rest_scale_quant_bits);
    Json::Value easy_rest_sh_meta;
    if (options.hybrid_easy_export_sh_drop) {
        const auto easy_bits = bitpack_utils::minimumBitsForValue(static_cast<std::uint32_t>(std::max(0, frozen.max_sh_degree)));
        bitpack_utils::writePackedUnsignedTensor(
            result_dir / "easy_rest_sh_levels.packed.bin",
            easy_export_sh_levels.to(torch::kInt32),
            easy_bits);
        easy_rest_sh_meta["storage"] = "packed_uint";
        easy_rest_sh_meta["bits"] = easy_bits;
        easy_rest_sh_meta["shape"] = tensorShapeJson(easy_export_sh_levels);
        easy_rest_sh_meta["max_value"] = frozen.max_sh_degree;
    }

    const auto full_rest_block_bases = blockMeansFromPerPointBase(
        frozen.features_rest_base.to(torch::kFloat32),
        frozen.block_ids,
        frozen.num_blocks,
        false).to(torch::kFloat16);
    const auto hard_rest_block_bases = full_rest_block_bases.index_select(0, frozen.hard_block_ids.to(torch::kLong));
    writeTensorBinary<c10::Half>(
        result_dir / "hard_features_rest_block_bases.bin",
        hard_rest_block_bases);

    Json::Value field_checkpoint_meta = saveFieldCheckpointForCompact(
        frozen,
        options,
        checkpoint_path,
        result_dir / "field_weights.pt");

    Json::Value phase2_field;
    phase2_field["num_fourier_frequencies"] = options.num_fourier_frequencies;
    phase2_field["use_hashgrid_encoder"] = options.use_hashgrid_encoder;
    phase2_field["hashgrid_num_levels"] = options.hashgrid_num_levels;
    phase2_field["hashgrid_features_per_level"] = options.hashgrid_features_per_level;
    phase2_field["hashgrid_log2_hashmap_size"] = options.hashgrid_log2_hashmap_size;
    phase2_field["hashgrid_base_resolution"] = options.hashgrid_base_resolution;
    phase2_field["hashgrid_per_level_scale"] = options.hashgrid_per_level_scale;
    phase2_field["hidden_dim"] = options.hidden_dim;
    phase2_field["num_hidden_layers"] = options.num_hidden_layers;
    phase2_field["batch_size"] = options.batch_size;
    phase2_field["max_steps"] = options.max_steps;
    phase2_field["log_interval"] = options.log_interval;
    phase2_field["eval_interval"] = options.eval_interval;
    phase2_field["learning_rate"] = options.learning_rate;
    phase2_field["weight_decay"] = options.weight_decay;
    phase2_field["include_features_dc"] = options.include_features_dc;
    phase2_field["include_opacity"] = options.include_opacity;
    phase2_field["include_scaling"] = options.include_scaling;
    phase2_field["include_rotation"] = options.include_rotation;
    phase2_field["predict_opacity"] = options.predict_opacity;
    phase2_field["predict_scaling"] = options.predict_scaling;
    phase2_field["predict_rotation"] = options.predict_rotation;
    phase2_field["block_embedding_dim"] = options.block_embedding_dim;
    phase2_field["hybrid_hard_only"] = options.hybrid_hard_only;
    phase2_field["hybrid_override_rest_only"] = options.hybrid_override_rest_only;
    phase2_field["hybrid_easy_export_sh_drop"] = options.hybrid_easy_export_sh_drop;
    phase2_field["hybrid_easy_export_sh_preserve_blocks"] = options.hybrid_easy_export_sh_preserve_blocks;
    phase2_field["hybrid_easy_export_sh_energy_keep_ratio"] = options.hybrid_easy_export_sh_energy_keep_ratio;
    phase2_field["hybrid_easy_export_sh_min_opacity"] = options.hybrid_easy_export_sh_min_opacity;
    phase2_field["hybrid_easy_export_sh_min_level"] = options.hybrid_easy_export_sh_min_level;
    phase2_field["phase2_compact_store_field_fp16"] = options.phase2_compact_store_field_fp16;
    phase2_field["phase2_compact_opacity_quant_bits"] = options.phase2_compact_opacity_quant_bits;
    phase2_field["phase2_compact_scaling_quant_bits"] = options.phase2_compact_scaling_quant_bits;
    phase2_field["phase2_compact_easy_rest_base_quant_bits"] = options.phase2_compact_easy_rest_base_quant_bits;
    phase2_field["phase2_compact_easy_rest_scale_quant_bits"] = options.phase2_compact_easy_rest_scale_quant_bits;
    phase2_field["phase2_compact_easy_rest_int2_rel_mse_threshold"] = options.phase2_compact_easy_rest_int2_rel_mse_threshold;
    phase2_field["phase2_compact_easy_rest_zlib"] = options.phase2_compact_easy_rest_zlib;
    phase2_field["phase2_compact_easy_rest_zlib_level"] = options.phase2_compact_easy_rest_zlib_level;
    phase2_field["phase2_compact_quantized_tensor_zlib"] = options.phase2_compact_quantized_tensor_zlib;
    phase2_field["phase2_compact_quantized_tensor_zlib_level"] = options.phase2_compact_quantized_tensor_zlib_level;
    phase2_field["phase2_compact_geometry_zlib"] = options.phase2_compact_geometry_zlib;
    phase2_field["phase2_compact_geometry_zlib_level"] = options.phase2_compact_geometry_zlib_level;
    phase2_field["phase2_compact_field_zlib"] = options.phase2_compact_field_zlib;
    phase2_field["phase2_compact_field_zlib_level"] = options.phase2_compact_field_zlib_level;
    phase2_field["phase2_compact_use_xz"] = options.phase2_compact_use_xz;

    Json::Value root;
    root["format"] = "phase2_residual_field_compact";
    root["representation"] = "selective_hybrid_rest_v1";
    root["field_checkpoint"] = "field_weights.pt";
    root["field_checkpoint_storage"] = field_checkpoint_meta;
    root["scene_extent"] = frozen.scene_extent;
    root["num_points"] = Json::Value::Int64(frozen.xyz.size(0));
    root["max_sh_degree"] = frozen.max_sh_degree;
    root["active_sh_degree"] = frozen.active_sh_degree;
    root["use_locality_base"] = frozen.use_locality_base;
    root["locality_high_sh_block_size"] = frozen.locality_high_sh_block_size;
    root["locality_low_sh_block_size"] = frozen.locality_low_sh_block_size;
    root["num_blocks"] = Json::Value::Int64(frozen.num_blocks);
    root["num_hard_blocks"] = Json::Value::Int64(frozen.num_hard_blocks);
    root["num_hard_points"] = Json::Value::Int64(frozen.num_hard_points);
    root["num_easy_points"] = Json::Value::Int64(easy_indices.numel());
    root["xyz"] = xyz_meta;
    root["features_dc"] = fdc_meta;
    root["sh_levels"] = sh_levels_meta;
    root["opacity"] = opacity_meta;
    root["scaling"] = scaling_meta;
    root["rotation"] = rotation_meta;
    root["hard_point_flags_storage"] = "packed_uint";
    root["hard_point_flags_bits"] = 1;
    root["hard_point_flags_shape"] = tensorShapeJson(frozen.hard_point_flags);
    root["hard_block_ids_shape"] = tensorShapeJson(frozen.hard_block_ids);
    root["hard_features_rest_block_bases_shape"] = tensorShapeJson(hard_rest_block_bases);
    root["easy_rest"] = easy_rest_meta;
    if (!easy_rest_sh_meta.isNull())
        root["easy_rest_sh_levels"] = easy_rest_sh_meta;
    root["phase2_field"] = phase2_field;

    writeMetadata(result_dir / "metadata.json", root);
}

DecodedGaussianTensors loadPhase2Compact(
    const std::filesystem::path& input_path,
    torch::DeviceType device_type)
{
    const auto result_dir = std::filesystem::is_directory(input_path) ? input_path : input_path.parent_path();
    const auto metadata_path = std::filesystem::is_directory(input_path) ? (input_path / "metadata.json") : input_path;
    const auto meta = readMetadataJson(metadata_path);
    if (meta.get("format", "").asString() != "phase2_residual_field_compact")
        throw std::runtime_error("Unsupported Phase 2 compact format at " + metadata_path.string());

    if (meta.get("representation", "").asString() == "selective_hybrid_rest_v1") {
        FrozenResidualFieldPackage frozen;
        frozen.max_sh_degree = meta["max_sh_degree"].asInt();
        frozen.active_sh_degree = meta["active_sh_degree"].asInt();
        frozen.scene_extent = meta["scene_extent"].asFloat();
        frozen.use_locality_base = meta.get("use_locality_base", true).asBool();
        frozen.locality_high_sh_block_size = meta.get("locality_high_sh_block_size", 64).asInt();
        frozen.locality_low_sh_block_size = meta.get("locality_low_sh_block_size", 128).asInt();
        frozen.num_blocks = meta.get("num_blocks", 0).asInt64();
        frozen.num_hard_blocks = meta.get("num_hard_blocks", 0).asInt64();
        frozen.num_hard_points = meta.get("num_hard_points", 0).asInt64();

        const auto& xyz_meta = meta["xyz"];
        const auto xyz_storage = xyz_meta.get("storage", xyz_meta.get("codec", "fp16").asString()).asString();
        if (xyz_storage == "morton_delta_varint") {
            auto geom_path = materializeMaybeCompressedFile(
                result_dir / "xyz.geom",
                xyz_meta.get("file_storage", Json::Value(Json::nullValue)),
                "xyz_geom");
            frozen.xyz = geometry_codec::decodeMortonDelta(geom_path, xyz_meta, device_type).to(torch::kFloat32);
            if (geom_path != result_dir / "xyz.geom")
                std::filesystem::remove(geom_path);
        } else
            frozen.xyz = readTensorBinary<c10::Half>(
                result_dir / "xyz.bin",
                tensorShapeFromJson(xyz_meta["shape"]),
                torch::kFloat16,
                device_type).to(torch::kFloat32);
        frozen.xyz_normalized = normalizeFromBounds(frozen.xyz, xyz_meta["bbox_min"], xyz_meta["bbox_max"]);

        frozen.features_dc = loadQuantizedTensorUint(
            result_dir / "features_dc.bin",
            meta["features_dc"],
            device_type).to(torch::kFloat32);
        frozen.features_dc = sanitizeTensorFinite(frozen.features_dc);

        const auto& sh_meta = meta["sh_levels"];
        const auto sh_shape = tensorShapeFromJson(sh_meta["shape"]);
        const auto sh_count = elementCount(sh_shape);
        if (sh_meta.get("storage", "").asString() == "packed_uint") {
            frozen.sh_levels = bitpack_utils::readPackedUnsignedTensor(
                result_dir / "sh_levels.packed.bin",
                sh_count,
                static_cast<std::uint8_t>(sh_meta["bits"].asUInt()),
                device_type).view(sh_shape).to(torch::kInt32);
        }
        else {
            frozen.sh_levels = readTensorBinary<int32_t>(
                result_dir / "sh_levels.bin",
                sh_shape,
                torch::kInt32,
                device_type);
        }
        frozen.block_ids = locality_codec::computeRestBlockIds(
            frozen.sh_levels,
            frozen.max_sh_degree,
            localityBaseExportOptions(frozen));
        if (frozen.num_blocks <= 0 && frozen.block_ids.defined() && frozen.block_ids.numel() > 0)
            frozen.num_blocks = frozen.block_ids.max().item<int64_t>() + 1;

        frozen.opacity = loadQuantizedTensorUint(
            result_dir / "opacity.bin",
            meta["opacity"],
            device_type).to(torch::kFloat32);
        if (meta["opacity"].get("representation", "activation").asString() == "activation")
            frozen.opacity = general_utils::inverse_sigmoid(torch::clamp(frozen.opacity, 1e-6f, 1.0f - 1e-6f));
        frozen.opacity = sanitizeOpacityLogits(frozen.opacity);

        frozen.scaling = loadQuantizedTensorUint(
            result_dir / "scaling.bin",
            meta["scaling"],
            device_type).to(torch::kFloat32);
        frozen.scaling = sanitizeTensorFinite(frozen.scaling);

        frozen.rotation = loadQuantizedTensorUint(
            result_dir / "rotation.bin",
            meta["rotation"],
            device_type).to(torch::kFloat32);
        frozen.rotation = maybeNormalizeQuaternionRows(flattenTensor(sanitizeTensorFinite(frozen.rotation))).view_as(frozen.rotation);

        frozen.opacity_base = frozen.opacity.clone();
        frozen.scaling_base = frozen.scaling.clone();
        frozen.rotation_base = frozen.rotation.clone();

        frozen.hard_point_flags = bitpack_utils::readPackedUnsignedTensor(
            result_dir / "hard_point_flags.packed.bin",
            elementCount(tensorShapeFromJson(meta["hard_point_flags_shape"])),
            static_cast<std::uint8_t>(meta.get("hard_point_flags_bits", 1).asUInt()),
            device_type).view(tensorShapeFromJson(meta["hard_point_flags_shape"])).to(torch::kInt32);
        frozen.hard_block_ids = readTensorBinary<int64_t>(
            result_dir / "hard_block_ids.bin",
            tensorShapeFromJson(meta["hard_block_ids_shape"]),
            torch::kLong,
            device_type);

        const auto easy_indices = indicesFromPackedFlags(frozen.hard_point_flags, false).to(device_type);
        auto easy_sh_levels = frozen.sh_levels.index_select(0, easy_indices);
        if (meta.isMember("easy_rest_sh_levels")) {
            const auto& easy_sh_meta = meta["easy_rest_sh_levels"];
            const auto easy_sh_shape = tensorShapeFromJson(easy_sh_meta["shape"]);
            const auto easy_sh_count = elementCount(easy_sh_shape);
            if (easy_sh_meta.get("storage", "").asString() == "packed_uint") {
                easy_sh_levels = bitpack_utils::readPackedUnsignedTensor(
                    result_dir / "easy_rest_sh_levels.packed.bin",
                    easy_sh_count,
                    static_cast<std::uint8_t>(easy_sh_meta["bits"].asUInt()),
                    device_type).view(easy_sh_shape).to(torch::kInt32);
            }
        }
        const auto easy_encoded = loadEncodedRestPayload(result_dir, "easy_rest", meta["easy_rest"]);
        const auto easy_payload = locality_codec::decodeRestPayload(
            easy_encoded,
            easy_sh_levels,
            easy_indices.numel(),
            frozen.max_sh_degree);
        auto easy_rest = restoreRestPayloadTensor(
            easy_payload,
            easy_sh_levels,
            easy_indices.numel(),
            frozen.max_sh_degree,
            device_type);

        frozen.features_rest_target = torch::zeros(
            {frozen.xyz.size(0), std::max(0, (frozen.max_sh_degree + 1) * (frozen.max_sh_degree + 1) - 1), 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(frozen.xyz.device()));
        frozen.features_rest_target.index_copy_(0, easy_indices, easy_rest);

        auto hard_rest_block_bases = readTensorBinary<c10::Half>(
            result_dir / "hard_features_rest_block_bases.bin",
            tensorShapeFromJson(meta["hard_features_rest_block_bases_shape"]),
            torch::kFloat16,
            device_type).to(torch::kFloat32);
        frozen.features_rest_base = expandHardBlockBasesToPerPoint(
            hard_rest_block_bases,
            frozen.block_ids,
            frozen.hard_point_flags,
            frozen.hard_block_ids,
            frozen.xyz.size(0)).to(torch::kFloat32);

        auto train_options = trainOptionsFromJson(meta["phase2_field"]);
        Phase2ResidualField model(frozen.max_sh_degree, frozen.num_blocks, train_options);
        model->to(frozen.xyz_normalized.device());
        torch::serialize::InputArchive archive;
        auto field_checkpoint_path = materializeMaybeCompressedFile(
            result_dir / meta.get("field_checkpoint", "field_weights.pt").asString(),
            meta.get("field_checkpoint_storage", Json::Value(Json::nullValue)),
            "field_ckpt");
        archive.load_from(field_checkpoint_path.string());
        model->load(archive);
        model->to(frozen.xyz_normalized.device(), torch::kFloat32);
        if (field_checkpoint_path != result_dir / meta.get("field_checkpoint", "field_weights.pt").asString())
            std::filesystem::remove(field_checkpoint_path);

        auto prediction_mask = buildPredictionMask(frozen, train_options);
        auto predicted_flat = fullInference(model, frozen, train_options, prediction_mask, std::max(32768, train_options.batch_size));
        return buildDecodedFromPrediction(frozen, predicted_flat, train_options);
    }

    if (meta.get("representation", "").asString() == "anchors_plus_field_weights") {
        FrozenResidualFieldPackage frozen;
        frozen.max_sh_degree = meta["max_sh_degree"].asInt();
        frozen.active_sh_degree = meta["active_sh_degree"].asInt();
        frozen.scene_extent = meta["scene_extent"].asFloat();
        frozen.use_locality_base = meta.get("use_locality_base", true).asBool();
        frozen.locality_high_sh_block_size = meta.get("locality_high_sh_block_size", 64).asInt();
        frozen.locality_low_sh_block_size = meta.get("locality_low_sh_block_size", 128).asInt();
        frozen.num_blocks = meta.get("num_blocks", 0).asInt64();

        frozen.xyz = readTensorBinary<c10::Half>(
            result_dir / "xyz.bin",
            tensorShapeFromJson(meta["xyz_shape"]),
            torch::kFloat16,
            device_type).to(torch::kFloat32);
        frozen.xyz_normalized = readTensorBinary<c10::Half>(
            result_dir / "xyz_normalized.bin",
            tensorShapeFromJson(meta["xyz_normalized_shape"]),
            torch::kFloat16,
            device_type).to(torch::kFloat32);
        frozen.features_dc = readTensorBinary<c10::Half>(
            result_dir / "features_dc.bin",
            tensorShapeFromJson(meta["features_dc_shape"]),
            torch::kFloat16,
            device_type).to(torch::kFloat32);
        frozen.opacity = readTensorBinary<c10::Half>(
            result_dir / "opacity.bin",
            tensorShapeFromJson(meta["opacity_shape"]),
            torch::kFloat16,
            device_type).to(torch::kFloat32);
        frozen.scaling = readTensorBinary<c10::Half>(
            result_dir / "scaling.bin",
            tensorShapeFromJson(meta["scaling_shape"]),
            torch::kFloat16,
            device_type).to(torch::kFloat32);
        frozen.rotation = readTensorBinary<c10::Half>(
            result_dir / "rotation.bin",
            tensorShapeFromJson(meta["rotation_shape"]),
            torch::kFloat16,
            device_type).to(torch::kFloat32);
        frozen.sh_levels = readTensorBinary<int32_t>(
            result_dir / "sh_levels.bin",
            tensorShapeFromJson(meta["sh_levels_shape"]),
            torch::kInt32,
            device_type);
        frozen.block_ids = locality_codec::computeRestBlockIds(
            frozen.sh_levels,
            frozen.max_sh_degree,
            localityBaseExportOptions(frozen));
        if (frozen.num_blocks <= 0 && frozen.block_ids.defined() && frozen.block_ids.numel() > 0)
            frozen.num_blocks = frozen.block_ids.max().item<int64_t>() + 1;

        auto block_bases = readTensorBinary<c10::Half>(
            result_dir / "features_rest_block_bases.bin",
            tensorShapeFromJson(meta["features_rest_block_bases_shape"]),
            torch::kFloat16,
            device_type).to(torch::kFloat32);
        frozen.features_rest_base = locality_codec::expandRestBlockBases(
            block_bases,
            frozen.sh_levels,
            frozen.max_sh_degree,
            localityBaseExportOptions(frozen)).to(torch::kFloat32);
        frozen.features_rest_target = torch::zeros_like(frozen.features_rest_base);
        frozen.opacity = sanitizeOpacityLogits(frozen.opacity);
        frozen.scaling = sanitizeTensorFinite(frozen.scaling);
        frozen.rotation = sanitizeTensorFinite(frozen.rotation);
        frozen.opacity_base = frozen.opacity;
        frozen.scaling_base = frozen.scaling;
        frozen.rotation_base = maybeNormalizeQuaternionRows(flattenTensor(frozen.rotation)).view_as(frozen.rotation);

        auto train_options = trainOptionsFromJson(meta["phase2_field"]);
        Phase2ResidualField model(frozen.max_sh_degree, frozen.num_blocks, train_options);
        model->to(frozen.xyz_normalized.device());
        torch::serialize::InputArchive archive;
        auto field_checkpoint_path = materializeMaybeCompressedFile(
            result_dir / meta.get("field_checkpoint", "field_weights.pt").asString(),
            meta.get("field_checkpoint_storage", Json::Value(Json::nullValue)),
            "field_ckpt");
        archive.load_from(field_checkpoint_path.string());
        model->load(archive);
        model->to(frozen.xyz_normalized.device(), torch::kFloat32);
        if (field_checkpoint_path != result_dir / meta.get("field_checkpoint", "field_weights.pt").asString())
            std::filesystem::remove(field_checkpoint_path);

        auto prediction_mask = buildPredictionMask(frozen, train_options);
        auto predicted_flat = fullInference(model, frozen, train_options, prediction_mask, std::max(32768, train_options.batch_size));
        return buildDecodedFromPrediction(frozen, predicted_flat, train_options);
    }

    FrozenResidualFieldPackage frozen;
    frozen.max_sh_degree = meta["max_sh_degree"].asInt();
    frozen.active_sh_degree = meta["active_sh_degree"].asInt();
    frozen.scene_extent = meta["scene_extent"].asFloat();
    frozen.use_locality_base = meta.get("use_locality_base", true).asBool();
    frozen.locality_high_sh_block_size = meta.get("locality_high_sh_block_size", 64).asInt();
    frozen.locality_low_sh_block_size = meta.get("locality_low_sh_block_size", 128).asInt();
    frozen.num_blocks = meta.get("num_blocks", 0).asInt64();

    const auto& xyz_meta = meta["xyz"];
    const auto xyz_storage = xyz_meta.get("storage", xyz_meta.get("codec", "fp16").asString()).asString();
    if (xyz_storage == "morton_delta_varint") {
        auto geom_path = materializeMaybeCompressedFile(
            result_dir / "xyz.geom",
            xyz_meta.get("file_storage", Json::Value(Json::nullValue)),
            "xyz_geom");
        frozen.xyz = geometry_codec::decodeMortonDelta(geom_path, xyz_meta, device_type).to(torch::kFloat32);
        if (geom_path != result_dir / "xyz.geom")
            std::filesystem::remove(geom_path);
    } else
        frozen.xyz = readTensorBinary<c10::Half>(
            result_dir / "xyz.bin",
            tensorShapeFromJson(xyz_meta["shape"]),
            torch::kFloat16,
            device_type).to(torch::kFloat32);
    frozen.xyz_normalized = normalizeFromBounds(frozen.xyz, xyz_meta["bbox_min"], xyz_meta["bbox_max"]);

    frozen.features_dc = loadQuantizedTensorUint(
        result_dir / "features_dc.bin",
        meta["features_dc"],
        device_type).to(torch::kFloat32);
    frozen.features_dc = sanitizeTensorFinite(frozen.features_dc);

    const auto& sh_meta = meta["sh_levels"];
    const auto sh_shape = tensorShapeFromJson(sh_meta["shape"]);
    const auto sh_count = elementCount(sh_shape);
    if (sh_meta.get("storage", "").asString() == "packed_uint") {
        frozen.sh_levels = bitpack_utils::readPackedUnsignedTensor(
            result_dir / "sh_levels.packed.bin",
            sh_count,
            static_cast<std::uint8_t>(sh_meta["bits"].asUInt()),
            device_type).view(sh_shape).to(torch::kInt32);
    }
    else {
        frozen.sh_levels = readTensorBinary<int32_t>(
            result_dir / "sh_levels.bin",
            sh_shape,
            torch::kInt32,
            device_type);
    }
    frozen.block_ids = locality_codec::computeRestBlockIds(
        frozen.sh_levels,
        frozen.max_sh_degree,
        localityBaseExportOptions(frozen));
    if (frozen.num_blocks <= 0 && frozen.block_ids.defined() && frozen.block_ids.numel() > 0)
        frozen.num_blocks = frozen.block_ids.max().item<int64_t>() + 1;

    auto rest_block_bases = readTensorBinary<c10::Half>(
        result_dir / "features_rest_block_bases.bin",
        tensorShapeFromJson(meta["features_rest_block_bases_shape"]),
        torch::kFloat16,
        device_type).to(torch::kFloat32);
    frozen.features_rest_base = locality_codec::expandRestBlockBases(
        rest_block_bases,
        frozen.sh_levels,
        frozen.max_sh_degree,
        localityBaseExportOptions(frozen)).to(torch::kFloat32);
    frozen.features_rest_target = torch::zeros_like(frozen.features_rest_base);
    frozen.opacity_base = torch::zeros({frozen.xyz.size(0), 1}, torch::TensorOptions().dtype(torch::kFloat32).device(frozen.xyz.device()));
    frozen.scaling_base = torch::zeros({frozen.xyz.size(0), 3}, torch::TensorOptions().dtype(torch::kFloat32).device(frozen.xyz.device()));
    frozen.rotation_base = torch::zeros({frozen.xyz.size(0), 4}, torch::TensorOptions().dtype(torch::kFloat32).device(frozen.xyz.device()));

    const auto opacity_storage = meta.get("opacity_storage", "field_predicted").asString();
    if (opacity_storage == "field_predicted") {
        auto opacity_block_bases = readTensorBinary<c10::Half>(
            result_dir / "opacity_block_bases.bin",
            tensorShapeFromJson(meta["opacity_block_bases_shape"]),
            torch::kFloat16,
            device_type).to(torch::kFloat32);
        frozen.opacity_base = locality_codec::expandBlockMeans(opacity_block_bases, frozen.block_ids).to(torch::kFloat32);
        frozen.opacity_base = sanitizeOpacityLogits(frozen.opacity_base);
        frozen.opacity = frozen.opacity_base.clone();
    }
    else {
        frozen.opacity = readTensorBinary<c10::Half>(
            result_dir / "opacity.bin",
            tensorShapeFromJson(meta["opacity_shape"]),
            torch::kFloat16,
            device_type).to(torch::kFloat32);
        frozen.opacity = sanitizeOpacityLogits(frozen.opacity);
        frozen.opacity_base = frozen.opacity.clone();
    }

    const auto scaling_storage = meta.get("scaling_storage", "field_predicted").asString();
    if (scaling_storage == "field_predicted") {
        auto scaling_block_bases = readTensorBinary<c10::Half>(
            result_dir / "scaling_block_bases.bin",
            tensorShapeFromJson(meta["scaling_block_bases_shape"]),
            torch::kFloat16,
            device_type).to(torch::kFloat32);
        frozen.scaling_base = locality_codec::expandBlockMeans(scaling_block_bases, frozen.block_ids).to(torch::kFloat32);
        frozen.scaling_base = sanitizeTensorFinite(frozen.scaling_base);
        frozen.scaling = frozen.scaling_base.clone();
    }
    else {
        frozen.scaling = readTensorBinary<c10::Half>(
            result_dir / "scaling.bin",
            tensorShapeFromJson(meta["scaling_shape"]),
            torch::kFloat16,
            device_type).to(torch::kFloat32);
        frozen.scaling = sanitizeTensorFinite(frozen.scaling);
        frozen.scaling_base = frozen.scaling.clone();
    }

    const auto rotation_storage = meta.get("rotation_storage", "field_predicted").asString();
    if (rotation_storage == "field_predicted") {
        auto rotation_block_bases = readTensorBinary<c10::Half>(
            result_dir / "rotation_block_bases.bin",
            tensorShapeFromJson(meta["rotation_block_bases_shape"]),
            torch::kFloat16,
            device_type).to(torch::kFloat32);
        frozen.rotation_base = locality_codec::expandBlockMeans(rotation_block_bases, frozen.block_ids).to(torch::kFloat32);
        frozen.rotation_base = maybeNormalizeQuaternionRows(flattenTensor(frozen.rotation_base)).view_as(frozen.rotation_base);
        frozen.rotation = frozen.rotation_base.clone();
    }
    else {
        frozen.rotation = readTensorBinary<c10::Half>(
            result_dir / "rotation.bin",
            tensorShapeFromJson(meta["rotation_shape"]),
            torch::kFloat16,
            device_type).to(torch::kFloat32);
        frozen.rotation = sanitizeTensorFinite(frozen.rotation);
        frozen.rotation_base = maybeNormalizeQuaternionRows(flattenTensor(frozen.rotation)).view_as(frozen.rotation);
    }

    auto train_options = trainOptionsFromJson(meta["phase2_field"]);
    Phase2ResidualField model(frozen.max_sh_degree, frozen.num_blocks, train_options);
    model->to(frozen.xyz_normalized.device());
    torch::serialize::InputArchive archive;
    auto field_checkpoint_path = materializeMaybeCompressedFile(
        result_dir / meta.get("field_checkpoint", "field_weights.pt").asString(),
        meta.get("field_checkpoint_storage", Json::Value(Json::nullValue)),
        "field_ckpt");
    archive.load_from(field_checkpoint_path.string());
    model->load(archive);
    model->to(frozen.xyz_normalized.device(), torch::kFloat32);
    if (field_checkpoint_path != result_dir / meta.get("field_checkpoint", "field_weights.pt").asString())
        std::filesystem::remove(field_checkpoint_path);

    auto prediction_mask = buildPredictionMask(frozen, train_options);
    auto predicted_flat = fullInference(model, frozen, train_options, prediction_mask, std::max(32768, train_options.batch_size));
    return buildDecodedFromPrediction(frozen, predicted_flat, train_options);
}

Phase2ResidualFieldTrainResult trainResidualField(
    const FrozenResidualFieldPackage& frozen,
    const Phase2ResidualFieldTrainOptions& options,
    const std::filesystem::path& result_dir)
{
    ensureDirectory(result_dir);
    ensureDirectory(result_dir / "checkpoints");
    ensureDirectory(result_dir / "predictions");

    Phase2ResidualField model(frozen.max_sh_degree, frozen.num_blocks, options);
    model->to(frozen.xyz_normalized.device());
    model->train();

    auto xyz_normalized = frozen.xyz_normalized.to(torch::kFloat32);
    auto features_dc = frozen.features_dc.to(torch::kFloat32);
    auto opacity_input = phase2InputOpacity(frozen, options).to(torch::kFloat32);
    auto scaling_input = phase2InputScaling(frozen, options).to(torch::kFloat32);
    auto rotation_input = phase2InputRotation(frozen, options).to(torch::kFloat32);
    auto sh_levels = frozen.sh_levels.to(torch::kInt32);
    auto target = buildTrainingTarget(frozen, options).to(torch::kFloat32);
    auto prediction_mask = buildPredictionMask(frozen, options).to(torch::kFloat32);
    auto masked_target = target * prediction_mask;

    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(options.learning_rate).weight_decay(options.weight_decay));

    const auto num_points = xyz_normalized.size(0);
    auto training_indices = torch::arange(
        num_points,
        torch::TensorOptions().dtype(torch::kLong).device(xyz_normalized.device()));
    if (options.hybrid_hard_only && frozen.hard_point_flags.defined() && frozen.hard_point_flags.numel() == num_points) {
        auto hard_indices = torch::nonzero(frozen.hard_point_flags.to(torch::kBool)).view({-1});
        if (hard_indices.numel() > 0)
            training_indices = hard_indices.to(xyz_normalized.device());
    }
    const auto train_point_count = training_indices.numel();
    const int batch_size = std::min<int64_t>(std::max(1, options.batch_size), train_point_count);
    float best_loss = std::numeric_limits<float>::max();
    float final_eval_loss = std::numeric_limits<float>::max();

    for (int step = 1; step <= options.max_steps; ++step) {
        auto batch_sample = torch::randint(
            train_point_count,
            {batch_size},
            torch::TensorOptions().dtype(torch::kLong).device(xyz_normalized.device()));
        auto batch_index = training_indices.index_select(0, batch_sample);

        auto prediction = model->forward(
            xyz_normalized.index_select(0, batch_index),
            sh_levels.index_select(0, batch_index),
            features_dc.index_select(0, batch_index),
            opacity_input.index_select(0, batch_index),
            scaling_input.index_select(0, batch_index),
            rotation_input.index_select(0, batch_index),
            frozen.block_ids.index_select(0, batch_index));
        prediction = prediction * prediction_mask.index_select(0, batch_index);
        auto batch_target = masked_target.index_select(0, batch_index);
        auto loss = torch::mse_loss(prediction, batch_target);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if (options.log_interval > 0 && (step == 1 || step % options.log_interval == 0 || step == options.max_steps))
            std::cout << "[Phase2ResidualField] step=" << step
                      << " train_mse=" << loss.item<float>() << std::endl;

        if (options.eval_interval > 0 && (step % options.eval_interval == 0 || step == options.max_steps)) {
            auto full_prediction = fullInference(model, frozen, options, prediction_mask, std::max(32768, batch_size));
            auto eval_loss = torch::mse_loss(full_prediction, masked_target).item<float>();
            final_eval_loss = eval_loss;
            std::cout << "[Phase2ResidualField] step=" << step
                      << " eval_mse=" << eval_loss << std::endl;
            if (eval_loss < best_loss) {
                best_loss = eval_loss;
                torch::serialize::OutputArchive archive;
                model->save(archive);
                archive.save_to((result_dir / "checkpoints/model_best.pt").string());
            }
        }
    }

    torch::serialize::OutputArchive last_archive;
    model->save(last_archive);
    last_archive.save_to((result_dir / "checkpoints/model_last.pt").string());

    auto predicted_flat = fullInference(model, frozen, options, prediction_mask, std::max(32768, batch_size));
    auto decoded = buildDecodedFromPrediction(frozen, predicted_flat, options);
    auto prediction_slices = splitPrediction(predicted_flat, frozen.max_sh_degree, options);
    auto predicted_features_rest_residual = prediction_slices.rest_residual.view_as(frozen.features_rest_target);

    writeTensorBinary<c10::Half>(
        result_dir / "predictions/features_rest_residual_pred.bin",
        predicted_features_rest_residual.to(torch::kFloat16));
    writeTensorBinary<c10::Half>(
        result_dir / "predictions/features_rest_pred.bin",
        decoded.features_rest.to(torch::kFloat16));

    Json::Value summary;
    summary["format"] = "phase2_residual_field_training";
    summary["num_points"] = Json::Value::Int64(num_points);
    summary["max_sh_degree"] = frozen.max_sh_degree;
    summary["active_sh_degree"] = frozen.active_sh_degree;
    summary["num_fourier_frequencies"] = options.num_fourier_frequencies;
    summary["use_hashgrid_encoder"] = options.use_hashgrid_encoder;
    summary["hashgrid_num_levels"] = options.hashgrid_num_levels;
    summary["hashgrid_features_per_level"] = options.hashgrid_features_per_level;
    summary["hidden_dim"] = options.hidden_dim;
    summary["num_hidden_layers"] = options.num_hidden_layers;
    summary["batch_size"] = batch_size;
    summary["max_steps"] = options.max_steps;
    summary["learning_rate"] = options.learning_rate;
    summary["weight_decay"] = options.weight_decay;
    summary["include_features_dc"] = options.include_features_dc;
    summary["include_opacity"] = options.include_opacity;
    summary["include_scaling"] = options.include_scaling;
    summary["include_rotation"] = options.include_rotation;
    summary["predict_opacity"] = options.predict_opacity;
    summary["predict_scaling"] = options.predict_scaling;
    summary["predict_rotation"] = options.predict_rotation;
    summary["block_embedding_dim"] = options.block_embedding_dim;
    summary["hybrid_hard_only"] = options.hybrid_hard_only;
    summary["hybrid_override_rest_only"] = options.hybrid_override_rest_only;
    summary["use_locality_base"] = frozen.features_rest_base.abs().sum().item<float>() > 0.0f;
    summary["target_type"] = frozen.features_rest_base.abs().sum().item<float>() > 0.0f ? "locality_residual" : "full_features_rest";
    summary["best_eval_mse"] = best_loss;
    summary["final_eval_mse"] = final_eval_loss;
    summary["trained_steps"] = options.max_steps;
    summary["train_point_count"] = Json::Value::Int64(train_point_count);
    summary["num_hard_blocks"] = Json::Value::Int64(frozen.num_hard_blocks);
    summary["num_hard_points"] = Json::Value::Int64(frozen.num_hard_points);
    summary["features_rest_base_shape"] = tensorShapeJson(frozen.features_rest_base);
    summary["block_ids_shape"] = tensorShapeJson(frozen.block_ids);
    summary["num_blocks"] = Json::Value::Int64(frozen.num_blocks);
    summary["predicted_features_rest_residual_shape"] = tensorShapeJson(predicted_features_rest_residual);
    summary["predicted_features_rest_shape"] = tensorShapeJson(decoded.features_rest);
    if (prediction_slices.opacity_residual.defined())
        summary["predicted_opacity_residual_shape"] = tensorShapeJson(prediction_slices.opacity_residual);
    if (prediction_slices.scaling_residual.defined())
        summary["predicted_scaling_residual_shape"] = tensorShapeJson(prediction_slices.scaling_residual);
    if (prediction_slices.rotation_residual.defined())
        summary["predicted_rotation_residual_shape"] = tensorShapeJson(prediction_slices.rotation_residual);
    writeMetadata(result_dir / "training_summary.json", summary);

    if (options.save_decoded_compact) {
        ensureDirectory(result_dir / "decoded_compact");
        ensureDirectory(result_dir / "decoded_compact" / "iteration_0");
        GaussianCodec::save(
            decoded,
            result_dir / "decoded_compact" / "iteration_0",
            highPrecisionDecodedExportOptions(options));
    }
    if (options.save_phase2_compact) {
        if (options.hybrid_hard_only) {
            ensureDirectory(result_dir / "phase2_compact");
            ensureDirectory(result_dir / "phase2_compact" / "iteration_0");
            saveSelectiveHybridPackage(
                frozen,
                decoded,
                options,
                result_dir / "checkpoints/model_best.pt",
                result_dir / "phase2_compact" / "iteration_0");
        }
        else {
            ensureDirectory(result_dir / "phase2_compact");
            ensureDirectory(result_dir / "phase2_compact" / "iteration_0");
            savePhase2CompactPackage(
                frozen,
                options,
                result_dir / "checkpoints/model_best.pt",
                result_dir / "phase2_compact" / "iteration_0");
        }
    }

    Phase2ResidualFieldTrainResult result;
    result.best_loss = best_loss;
    result.final_eval_loss = final_eval_loss;
    result.trained_steps = options.max_steps;
    result.decoded = decoded;
    return result;
}

} // namespace phase2_residual_field
