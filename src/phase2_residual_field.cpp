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

#include <json/json.h>
#include <torch/torch.h>

#include "include/attribute_sort.h"
#include "include/bitpack_utils.h"
#include "include/gaussian_codec.h"
#include "include/geometry_codec.h"
#include "include/locality_codec.h"
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
    if (root.isMember("phase2_compact_pack_sh_levels"))
        options.phase2_compact_pack_sh_levels = root["phase2_compact_pack_sh_levels"].asBool();
    if (root.isMember("phase2_compact_fdc_quant_bits"))
        options.phase2_compact_fdc_quant_bits = root["phase2_compact_fdc_quant_bits"].asInt();
    if (root.isMember("phase2_compact_use_geometry_codec"))
        options.phase2_compact_use_geometry_codec = root["phase2_compact_use_geometry_codec"].asBool();
    if (root.isMember("phase2_compact_geometry_quant_bits"))
        options.phase2_compact_geometry_quant_bits = root["phase2_compact_geometry_quant_bits"].asInt();
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

void saveQuantizedTensorUint(
    const std::filesystem::path& path,
    const torch::Tensor& tensor,
    int quant_bits,
    Json::Value& meta)
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

    const auto packed_bytes = bitpack_utils::packUnsignedValues(packed_values, static_cast<std::uint8_t>(bits));
    writeBinaryBytes(path, packed_bytes);

    meta["storage"] = "packed_uint";
    meta["bits"] = bits;
    meta["shape"] = tensorShapeJson(tensor);
    meta["mins"] = mins_json;
    meta["maxs"] = maxs_json;
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
    const auto packed_bytes = readBinaryBytes(path);
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
    auto rest_mask = buildRestMask(frozen.sh_levels, frozen.max_sh_degree).view({frozen.xyz.size(0), -1});
    std::vector<torch::Tensor> parts{rest_mask};
    auto ones_options = torch::TensorOptions().dtype(torch::kFloat32).device(rest_mask.device());
    if (options.predict_opacity)
        parts.push_back(torch::ones({rest_mask.size(0), 1}, ones_options));
    if (options.predict_scaling)
        parts.push_back(torch::ones({rest_mask.size(0), 3}, ones_options));
    if (options.predict_rotation)
        parts.push_back(torch::ones({rest_mask.size(0), 4}, ones_options));
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
    decoded.features_rest = (features_rest_base + slices.rest_residual.view_as(features_rest_base)).detach().to(torch::kFloat32);

    if (options.predict_opacity && slices.opacity_residual.defined()) {
        auto base = frozen.opacity_base.defined() ? frozen.opacity_base : torch::zeros_like(frozen.opacity);
        decoded.opacity = (flattenTensor(base) + slices.opacity_residual).view_as(base).detach().to(torch::kFloat32);
    }
    else {
        decoded.opacity = frozen.opacity.detach().to(torch::kFloat32);
    }

    if (options.predict_scaling && slices.scaling_residual.defined()) {
        auto base = frozen.scaling_base.defined() ? frozen.scaling_base : torch::zeros_like(frozen.scaling);
        decoded.scaling = (flattenTensor(base) + slices.scaling_residual).view_as(base).detach().to(torch::kFloat32);
    }
    else {
        decoded.scaling = frozen.scaling.detach().to(torch::kFloat32);
    }

    if (options.predict_rotation && slices.rotation_residual.defined()) {
        auto base = frozen.rotation_base.defined() ? frozen.rotation_base : torch::zeros_like(frozen.rotation);
        auto rotation = (flattenTensor(base) + slices.rotation_residual).view_as(base).detach().to(torch::kFloat32);
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
        fdc_meta);

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

    std::filesystem::copy_file(
        checkpoint_path,
        result_dir / "field_weights.pt",
        std::filesystem::copy_options::overwrite_existing);

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
    phase2_field["save_decoded_compact"] = options.save_decoded_compact;
    phase2_field["save_phase2_compact"] = options.save_phase2_compact;
    phase2_field["decoded_xyz_quant_bits"] = options.decoded_xyz_quant_bits;
    phase2_field["decoded_attribute_quant_bits"] = options.decoded_attribute_quant_bits;
    phase2_field["decoded_rotation_quant_bits"] = options.decoded_rotation_quant_bits;
    phase2_field["phase2_compact_pack_sh_levels"] = options.phase2_compact_pack_sh_levels;
    phase2_field["phase2_compact_fdc_quant_bits"] = options.phase2_compact_fdc_quant_bits;
    phase2_field["phase2_compact_use_geometry_codec"] = options.phase2_compact_use_geometry_codec;
    phase2_field["phase2_compact_geometry_quant_bits"] = options.phase2_compact_geometry_quant_bits;

    Json::Value root;
    root["format"] = "phase2_residual_field_compact";
    root["representation"] = "light_anchors_plus_field_weights_v3";
    root["field_checkpoint"] = "field_weights.pt";
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

DecodedGaussianTensors loadPhase2Compact(
    const std::filesystem::path& input_path,
    torch::DeviceType device_type)
{
    const auto result_dir = std::filesystem::is_directory(input_path) ? input_path : input_path.parent_path();
    const auto metadata_path = std::filesystem::is_directory(input_path) ? (input_path / "metadata.json") : input_path;
    const auto meta = readMetadataJson(metadata_path);
    if (meta.get("format", "").asString() != "phase2_residual_field_compact")
        throw std::runtime_error("Unsupported Phase 2 compact format at " + metadata_path.string());

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
        archive.load_from((result_dir / meta.get("field_checkpoint", "field_weights.pt").asString()).string());
        model->load(archive);

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
    if (xyz_storage == "morton_delta_varint")
        frozen.xyz = geometry_codec::decodeMortonDelta(result_dir / "xyz.geom", xyz_meta, device_type).to(torch::kFloat32);
    else
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
    archive.load_from((result_dir / meta.get("field_checkpoint", "field_weights.pt").asString()).string());
    model->load(archive);

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

    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(options.learning_rate).weight_decay(options.weight_decay));

    const auto num_points = xyz_normalized.size(0);
    const int batch_size = std::min<int64_t>(std::max(1, options.batch_size), num_points);
    float best_loss = std::numeric_limits<float>::max();
    float final_eval_loss = std::numeric_limits<float>::max();

    for (int step = 1; step <= options.max_steps; ++step) {
        auto batch_index = torch::randint(
            num_points,
            {batch_size},
            torch::TensorOptions().dtype(torch::kLong).device(xyz_normalized.device()));

        auto prediction = model->forward(
            xyz_normalized.index_select(0, batch_index),
            sh_levels.index_select(0, batch_index),
            features_dc.index_select(0, batch_index),
            opacity_input.index_select(0, batch_index),
            scaling_input.index_select(0, batch_index),
            rotation_input.index_select(0, batch_index),
            frozen.block_ids.index_select(0, batch_index));
        prediction = prediction * prediction_mask.index_select(0, batch_index);
        auto batch_target = target.index_select(0, batch_index);
        auto loss = torch::mse_loss(prediction, batch_target);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if (options.log_interval > 0 && (step == 1 || step % options.log_interval == 0 || step == options.max_steps))
            std::cout << "[Phase2ResidualField] step=" << step
                      << " train_mse=" << loss.item<float>() << std::endl;

        if (options.eval_interval > 0 && (step % options.eval_interval == 0 || step == options.max_steps)) {
            auto full_prediction = fullInference(model, frozen, options, prediction_mask, std::max(32768, batch_size));
            auto eval_loss = torch::mse_loss(full_prediction, target).item<float>();
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
    summary["use_locality_base"] = frozen.features_rest_base.abs().sum().item<float>() > 0.0f;
    summary["target_type"] = frozen.features_rest_base.abs().sum().item<float>() > 0.0f ? "locality_residual" : "full_features_rest";
    summary["best_eval_mse"] = best_loss;
    summary["final_eval_mse"] = final_eval_loss;
    summary["trained_steps"] = options.max_steps;
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
        ensureDirectory(result_dir / "phase2_compact");
        ensureDirectory(result_dir / "phase2_compact" / "iteration_0");
        savePhase2CompactPackage(
            frozen,
            options,
            result_dir / "checkpoints/model_best.pt",
            result_dir / "phase2_compact" / "iteration_0");
    }

    Phase2ResidualFieldTrainResult result;
    result.best_loss = best_loss;
    result.final_eval_loss = final_eval_loss;
    result.trained_steps = options.max_steps;
    result.decoded = decoded;
    return result;
}

} // namespace phase2_residual_field
