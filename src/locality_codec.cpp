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

#include "include/locality_codec.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

struct QuantizedResidualBlock
{
    std::uint8_t residual_bits = 8;
    std::vector<c10::Half> scales;
    std::vector<std::uint8_t> bytes;
    double relative_mse = 0.0;
};

std::vector<std::uint8_t> packInt4(const std::vector<int>& values)
{
    std::vector<std::uint8_t> packed((values.size() + 1) / 2, 0);
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
        const std::uint8_t nibble = static_cast<std::uint8_t>(std::clamp(values[idx] + 8, 0, 15));
        if ((idx & 1u) == 0u)
            packed[idx / 2] = nibble;
        else
            packed[idx / 2] |= static_cast<std::uint8_t>(nibble << 4);
    }
    return packed;
}

inline int unpackInt4(std::uint8_t packed, bool high)
{
    const std::uint8_t nibble = high ? static_cast<std::uint8_t>((packed >> 4) & 0xF) : static_cast<std::uint8_t>(packed & 0xF);
    return static_cast<int>(nibble) - 8;
}

QuantizedResidualBlock quantizeResidualBlock(
    const std::vector<float>& residuals,
    std::size_t point_count,
    std::size_t payload_dims,
    std::uint8_t residual_bits)
{
    QuantizedResidualBlock block;
    block.residual_bits = residual_bits;
    block.scales.resize(payload_dims, c10::Half(0.0f));

    if (point_count == 0 || payload_dims == 0 || residuals.empty())
        return block;

    const float qmax = residual_bits == 4 ? 7.0f : 127.0f;
    std::vector<float> scales(payload_dims, 0.0f);
    for (std::size_t dim = 0; dim < payload_dims; ++dim) {
        float max_abs = 0.0f;
        for (std::size_t point_idx = 0; point_idx < point_count; ++point_idx)
            max_abs = std::max(max_abs, std::abs(residuals[point_idx * payload_dims + dim]));
        scales[dim] = max_abs > 1e-8f ? (max_abs / qmax) : 0.0f;
        block.scales[dim] = c10::Half(scales[dim]);
    }

    double sum_sq = 0.0;
    double err_sq = 0.0;
    if (residual_bits == 4) {
        std::vector<int> quantized(residuals.size(), 0);
        for (std::size_t idx = 0; idx < residuals.size(); ++idx) {
            const std::size_t dim = idx % payload_dims;
            const float scale = scales[dim];
            int q = 0;
            if (scale > 0.0f)
                q = static_cast<int>(std::llround(static_cast<double>(residuals[idx]) / static_cast<double>(scale)));
            q = std::clamp(q, -8, 7);
            quantized[idx] = q;
            const float reconstructed = static_cast<float>(q) * scale;
            const double diff = static_cast<double>(residuals[idx] - reconstructed);
            err_sq += diff * diff;
            sum_sq += static_cast<double>(residuals[idx]) * static_cast<double>(residuals[idx]);
        }
        block.bytes = packInt4(quantized);
    }
    else {
        block.bytes.resize(residuals.size(), 0);
        for (std::size_t idx = 0; idx < residuals.size(); ++idx) {
            const std::size_t dim = idx % payload_dims;
            const float scale = scales[dim];
            int q = 0;
            if (scale > 0.0f)
                q = static_cast<int>(std::llround(static_cast<double>(residuals[idx]) / static_cast<double>(scale)));
            q = std::clamp(q, -127, 127);
            block.bytes[idx] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
            const float reconstructed = static_cast<float>(q) * scale;
            const double diff = static_cast<double>(residuals[idx] - reconstructed);
            err_sq += diff * diff;
            sum_sq += static_cast<double>(residuals[idx]) * static_cast<double>(residuals[idx]);
        }
    }

    block.relative_mse = err_sq / std::max(1e-12, sum_sq);
    return block;
}

std::size_t blockSizeForLevel(int sh_level, int max_sh_degree, const CompactExportOptions& options)
{
    if (sh_level >= max_sh_degree)
        return static_cast<std::size_t>(std::max(1, options.f_rest_locality_high_sh_block_size));
    return static_cast<std::size_t>(std::max(1, options.f_rest_locality_low_sh_block_size));
}

} // namespace

namespace locality_codec
{

std::size_t restPayloadValuesForLevel(int level)
{
    const int coeff_count = std::max(0, (level + 1) * (level + 1) - 1);
    return static_cast<std::size_t>(coeff_count * 3);
}

EncodedRestPayload encodeRestPayload(
    const torch::Tensor& features_rest,
    const torch::Tensor& sh_levels,
    int max_sh_degree,
    const CompactExportOptions& options)
{
    EncodedRestPayload encoded;
    if (max_sh_degree <= 0 || !features_rest.defined() || features_rest.numel() == 0)
        return encoded;

    auto rest_cpu = features_rest.detach().contiguous().to(torch::kCPU, torch::kFloat32);
    auto levels_cpu = sh_levels.detach().contiguous().to(torch::kCPU, torch::kInt32);

    const float* rest_ptr = rest_cpu.data_ptr<float>();
    const std::int32_t* levels_ptr = levels_cpu.data_ptr<std::int32_t>();
    const std::int64_t num_points = rest_cpu.size(0);
    const std::int64_t rest_coeffs = rest_cpu.size(1);

    std::int64_t point_idx = 0;
    while (point_idx < num_points) {
        const int sh_level = std::clamp(static_cast<int>(levels_ptr[point_idx]), 0, max_sh_degree);
        const std::size_t payload_dims = restPayloadValuesForLevel(sh_level);
        const std::size_t target_block_size = blockSizeForLevel(sh_level, max_sh_degree, options);

        std::size_t block_points = 1;
        while (point_idx + static_cast<std::int64_t>(block_points) < num_points
               && block_points < target_block_size
               && std::clamp(static_cast<int>(levels_ptr[point_idx + static_cast<std::int64_t>(block_points)]), 0, max_sh_degree) == sh_level) {
            ++block_points;
        }

        if (block_points > std::numeric_limits<std::uint16_t>::max())
            throw std::runtime_error("f_rest locality block exceeded uint16 point-count capacity.");

        RestBlockInfo block_info;
        block_info.sh_level = static_cast<std::uint8_t>(sh_level);
        block_info.point_count = static_cast<std::uint16_t>(block_points);

        if (payload_dims == 0) {
            block_info.residual_bits = 8;
            encoded.blocks.push_back(block_info);
            point_idx += static_cast<std::int64_t>(block_points);
            continue;
        }

        std::vector<float> base(payload_dims, 0.0f);
        for (std::size_t local_idx = 0; local_idx < block_points; ++local_idx) {
            const float* point_ptr = rest_ptr + ((point_idx + static_cast<std::int64_t>(local_idx)) * rest_coeffs * 3);
            for (std::size_t dim = 0; dim < payload_dims; ++dim)
                base[dim] += point_ptr[dim];
        }
        for (float& value : base)
            value /= static_cast<float>(block_points);

        std::vector<float> residuals(block_points * payload_dims, 0.0f);
        for (std::size_t local_idx = 0; local_idx < block_points; ++local_idx) {
            const float* point_ptr = rest_ptr + ((point_idx + static_cast<std::int64_t>(local_idx)) * rest_coeffs * 3);
            for (std::size_t dim = 0; dim < payload_dims; ++dim)
                residuals[local_idx * payload_dims + dim] = point_ptr[dim] - base[dim];
        }

        const auto int4_candidate = quantizeResidualBlock(residuals, block_points, payload_dims, 4);
        const auto int8_candidate = quantizeResidualBlock(residuals, block_points, payload_dims, 8);
        const bool use_int4 = int4_candidate.relative_mse <= static_cast<double>(options.f_rest_locality_int4_rel_mse_threshold);
        const auto& selected = use_int4 ? int4_candidate : int8_candidate;

        block_info.residual_bits = selected.residual_bits;
        encoded.blocks.push_back(block_info);
        encoded.payload_values += block_points * payload_dims;
        if (use_int4)
            ++encoded.int4_block_count;
        else
            ++encoded.int8_block_count;

        encoded.base_values.reserve(encoded.base_values.size() + payload_dims);
        encoded.scale_values.reserve(encoded.scale_values.size() + payload_dims);
        for (std::size_t dim = 0; dim < payload_dims; ++dim) {
            encoded.base_values.push_back(c10::Half(base[dim]));
            encoded.scale_values.push_back(selected.scales[dim]);
        }
        encoded.residual_bytes.insert(
            encoded.residual_bytes.end(),
            selected.bytes.begin(),
            selected.bytes.end());

        point_idx += static_cast<std::int64_t>(block_points);
    }

    return encoded;
}

torch::Tensor computeRestBlockBases(
    const torch::Tensor& features_rest,
    const torch::Tensor& sh_levels,
    int max_sh_degree,
    const CompactExportOptions& options)
{
    if (max_sh_degree <= 0 || !features_rest.defined() || features_rest.numel() == 0)
        return torch::zeros({0}, features_rest.options());

    auto rest_cpu = features_rest.detach().contiguous().to(torch::kCPU, torch::kFloat32);
    auto levels_cpu = sh_levels.detach().contiguous().to(torch::kCPU, torch::kInt32);

    const float* rest_ptr = rest_cpu.data_ptr<float>();
    const std::int32_t* levels_ptr = levels_cpu.data_ptr<std::int32_t>();
    const std::int64_t num_points = rest_cpu.size(0);
    const std::int64_t rest_coeffs = rest_cpu.size(1);
    const std::int64_t dims_per_point = rest_coeffs * 3;

    std::size_t num_blocks = 0;
    std::int64_t point_idx = 0;
    while (point_idx < num_points) {
        const int sh_level = std::clamp(static_cast<int>(levels_ptr[point_idx]), 0, max_sh_degree);
        const std::size_t target_block_size = blockSizeForLevel(sh_level, max_sh_degree, options);

        std::size_t block_points = 1;
        while (point_idx + static_cast<std::int64_t>(block_points) < num_points
               && block_points < target_block_size
               && std::clamp(static_cast<int>(levels_ptr[point_idx + static_cast<std::int64_t>(block_points)]), 0, max_sh_degree) == sh_level) {
            ++block_points;
        }
        ++num_blocks;
        point_idx += static_cast<std::int64_t>(block_points);
    }

    auto block_bases_cpu = torch::zeros(
        {static_cast<std::int64_t>(num_blocks), rest_coeffs, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    float* block_base_ptr = block_bases_cpu.data_ptr<float>();

    std::size_t block_idx = 0;
    std::int64_t point_offset = 0;
    while (point_offset < num_points) {
        const int sh_level = std::clamp(static_cast<int>(levels_ptr[point_offset]), 0, max_sh_degree);
        const std::size_t payload_dims = restPayloadValuesForLevel(sh_level);
        const std::size_t target_block_size = blockSizeForLevel(sh_level, max_sh_degree, options);

        std::size_t block_points = 1;
        while (point_offset + static_cast<std::int64_t>(block_points) < num_points
               && block_points < target_block_size
               && std::clamp(static_cast<int>(levels_ptr[point_offset + static_cast<std::int64_t>(block_points)]), 0, max_sh_degree) == sh_level) {
            ++block_points;
        }

        if (payload_dims > 0) {
            std::vector<float> base(payload_dims, 0.0f);
            for (std::size_t local_idx = 0; local_idx < block_points; ++local_idx) {
                const float* point_ptr =
                    rest_ptr + ((point_offset + static_cast<std::int64_t>(local_idx)) * dims_per_point);
                for (std::size_t dim = 0; dim < payload_dims; ++dim)
                    base[dim] += point_ptr[dim];
            }
            for (float& value : base)
                value /= static_cast<float>(block_points);

            float* block_ptr = block_base_ptr + (static_cast<std::int64_t>(block_idx) * dims_per_point);
            for (std::size_t dim = 0; dim < payload_dims; ++dim)
                block_ptr[dim] = base[dim];
        }

        ++block_idx;
        point_offset += static_cast<std::int64_t>(block_points);
    }

    return block_bases_cpu.to(features_rest.device());
}

torch::Tensor expandRestBlockBases(
    const torch::Tensor& block_bases,
    const torch::Tensor& sh_levels,
    int max_sh_degree,
    const CompactExportOptions& options)
{
    if (max_sh_degree <= 0 || !sh_levels.defined() || sh_levels.numel() == 0)
        return torch::zeros({0}, torch::TensorOptions().dtype(torch::kFloat32).device(sh_levels.device()));
    if (!block_bases.defined() || block_bases.numel() == 0)
        throw std::runtime_error("Cannot expand empty locality block bases.");

    auto levels_cpu = sh_levels.detach().contiguous().to(torch::kCPU, torch::kInt32);
    auto block_bases_cpu = block_bases.detach().contiguous().to(torch::kCPU, torch::kFloat32);
    const std::int32_t* levels_ptr = levels_cpu.data_ptr<std::int32_t>();
    const float* block_base_ptr = block_bases_cpu.data_ptr<float>();

    const std::int64_t num_points = levels_cpu.size(0);
    const std::int64_t rest_coeffs = block_bases_cpu.size(1);
    const std::int64_t dims_per_point = rest_coeffs * 3;

    auto expanded_cpu = torch::zeros(
        {num_points, rest_coeffs, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    float* expanded_ptr = expanded_cpu.data_ptr<float>();

    std::size_t block_idx = 0;
    std::int64_t point_idx = 0;
    while (point_idx < num_points) {
        const int sh_level = std::clamp(static_cast<int>(levels_ptr[point_idx]), 0, max_sh_degree);
        const std::size_t payload_dims = restPayloadValuesForLevel(sh_level);
        const std::size_t target_block_size = blockSizeForLevel(sh_level, max_sh_degree, options);

        std::size_t block_points = 1;
        while (point_idx + static_cast<std::int64_t>(block_points) < num_points
               && block_points < target_block_size
               && std::clamp(static_cast<int>(levels_ptr[point_idx + static_cast<std::int64_t>(block_points)]), 0, max_sh_degree) == sh_level) {
            ++block_points;
        }

        if (block_idx >= static_cast<std::size_t>(block_bases_cpu.size(0)))
            throw std::runtime_error("Locality block base tensor is truncated.");

        if (payload_dims > 0) {
            const float* block_ptr = block_base_ptr + (static_cast<std::int64_t>(block_idx) * dims_per_point);
            for (std::size_t local_idx = 0; local_idx < block_points; ++local_idx) {
                float* point_ptr = expanded_ptr + ((point_idx + static_cast<std::int64_t>(local_idx)) * dims_per_point);
                for (std::size_t dim = 0; dim < payload_dims; ++dim)
                    point_ptr[dim] = block_ptr[dim];
            }
        }

        ++block_idx;
        point_idx += static_cast<std::int64_t>(block_points);
    }

    if (block_idx != static_cast<std::size_t>(block_bases_cpu.size(0)))
        throw std::runtime_error("Locality block base tensor has extra blocks beyond point layout.");

    return expanded_cpu.to(block_bases.device());
}

torch::Tensor computeRestBlockIds(
    const torch::Tensor& sh_levels,
    int max_sh_degree,
    const CompactExportOptions& options)
{
    if (!sh_levels.defined() || sh_levels.numel() == 0)
        return torch::zeros({0}, torch::TensorOptions().dtype(torch::kLong).device(sh_levels.device()));

    auto levels_cpu = sh_levels.detach().contiguous().to(torch::kCPU, torch::kInt32);
    const std::int32_t* levels_ptr = levels_cpu.data_ptr<std::int32_t>();
    const std::int64_t num_points = levels_cpu.size(0);

    auto block_ids_cpu = torch::zeros(
        {num_points},
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
    auto* block_ids_ptr = block_ids_cpu.data_ptr<int64_t>();

    int64_t block_idx = 0;
    std::int64_t point_idx = 0;
    while (point_idx < num_points) {
        const int sh_level = std::clamp(static_cast<int>(levels_ptr[point_idx]), 0, max_sh_degree);
        const std::size_t target_block_size = blockSizeForLevel(sh_level, max_sh_degree, options);

        std::size_t block_points = 1;
        while (point_idx + static_cast<std::int64_t>(block_points) < num_points
               && block_points < target_block_size
               && std::clamp(static_cast<int>(levels_ptr[point_idx + static_cast<std::int64_t>(block_points)]), 0, max_sh_degree) == sh_level) {
            ++block_points;
        }

        for (std::size_t local_idx = 0; local_idx < block_points; ++local_idx)
            block_ids_ptr[point_idx + static_cast<std::int64_t>(local_idx)] = block_idx;

        ++block_idx;
        point_idx += static_cast<std::int64_t>(block_points);
    }

    return block_ids_cpu.to(sh_levels.device());
}

torch::Tensor computeRestBlockBase(
    const torch::Tensor& features_rest,
    const torch::Tensor& sh_levels,
    int max_sh_degree,
    const CompactExportOptions& options)
{
    if (max_sh_degree <= 0 || !features_rest.defined() || features_rest.numel() == 0)
        return torch::zeros_like(features_rest);
    auto block_bases = computeRestBlockBases(features_rest, sh_levels, max_sh_degree, options);
    return expandRestBlockBases(block_bases, sh_levels, max_sh_degree, options);
}

torch::Tensor computeBlockMeans(
    const torch::Tensor& values,
    const torch::Tensor& block_ids,
    int64_t num_blocks)
{
    if (!values.defined() || values.numel() == 0)
        return torch::zeros({0}, values.options());
    if (!block_ids.defined() || block_ids.numel() == 0 || num_blocks <= 0)
        throw std::runtime_error("computeBlockMeans expects non-empty block ids and positive block count.");
    if (values.size(0) != block_ids.size(0))
        throw std::runtime_error("computeBlockMeans: values and block_ids length mismatch.");

    auto values_cpu = values.detach().contiguous().to(torch::kCPU, torch::kFloat32);
    auto block_ids_cpu = block_ids.detach().contiguous().to(torch::kCPU, torch::kLong);
    const auto trailing_shape = values_cpu.sizes().vec();
    const int64_t flattened_dims = values_cpu.numel() / std::max<int64_t>(1, values_cpu.size(0));
    auto values_flat = values_cpu.view({values_cpu.size(0), flattened_dims});

    auto block_means = torch::zeros(
        {num_blocks, flattened_dims},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto counts = torch::zeros(
        {num_blocks},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    const float* values_ptr = values_flat.data_ptr<float>();
    const int64_t* block_ids_ptr = block_ids_cpu.data_ptr<int64_t>();
    float* means_ptr = block_means.data_ptr<float>();
    float* counts_ptr = counts.data_ptr<float>();

    for (int64_t point_idx = 0; point_idx < values_flat.size(0); ++point_idx) {
        const int64_t block_idx = block_ids_ptr[point_idx];
        if (block_idx < 0 || block_idx >= num_blocks)
            throw std::runtime_error("computeBlockMeans: block id out of range.");
        counts_ptr[block_idx] += 1.0f;
        for (int64_t dim = 0; dim < flattened_dims; ++dim)
            means_ptr[block_idx * flattened_dims + dim] += values_ptr[point_idx * flattened_dims + dim];
    }

    for (int64_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const float count = std::max(1.0f, counts_ptr[block_idx]);
        for (int64_t dim = 0; dim < flattened_dims; ++dim)
            means_ptr[block_idx * flattened_dims + dim] /= count;
    }

    std::vector<int64_t> output_shape;
    output_shape.reserve(static_cast<std::size_t>(values_cpu.dim()));
    output_shape.push_back(num_blocks);
    for (int dim_idx = 1; dim_idx < values_cpu.dim(); ++dim_idx)
        output_shape.push_back(values_cpu.size(dim_idx));
    return block_means.view(output_shape).to(values.device());
}

torch::Tensor expandBlockMeans(
    const torch::Tensor& block_means,
    const torch::Tensor& block_ids)
{
    if (!block_means.defined() || block_means.numel() == 0)
        throw std::runtime_error("expandBlockMeans expects non-empty block means.");
    if (!block_ids.defined() || block_ids.numel() == 0)
        return torch::zeros({0}, block_means.options());

    auto block_means_cpu = block_means.detach().contiguous().to(torch::kCPU, torch::kFloat32);
    auto block_ids_cpu = block_ids.detach().contiguous().to(torch::kCPU, torch::kLong);

    const int64_t flattened_dims = block_means_cpu.numel() / std::max<int64_t>(1, block_means_cpu.size(0));
    auto block_means_flat = block_means_cpu.view({block_means_cpu.size(0), flattened_dims});
    auto expanded = torch::zeros(
        {block_ids_cpu.size(0), flattened_dims},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    const float* means_ptr = block_means_flat.data_ptr<float>();
    const int64_t* block_ids_ptr = block_ids_cpu.data_ptr<int64_t>();
    float* expanded_ptr = expanded.data_ptr<float>();

    for (int64_t point_idx = 0; point_idx < block_ids_cpu.size(0); ++point_idx) {
        const int64_t block_idx = block_ids_ptr[point_idx];
        if (block_idx < 0 || block_idx >= block_means_flat.size(0))
            throw std::runtime_error("expandBlockMeans: block id out of range.");
        for (int64_t dim = 0; dim < flattened_dims; ++dim)
            expanded_ptr[point_idx * flattened_dims + dim] = means_ptr[block_idx * flattened_dims + dim];
    }

    std::vector<int64_t> output_shape;
    output_shape.reserve(static_cast<std::size_t>(block_means_cpu.dim()));
    output_shape.push_back(block_ids_cpu.size(0));
    for (int dim_idx = 1; dim_idx < block_means_cpu.dim(); ++dim_idx)
        output_shape.push_back(block_means_cpu.size(dim_idx));
    return expanded.view(output_shape).to(block_means.device());
}

torch::Tensor computeExpandedBlockMeans(
    const torch::Tensor& values,
    const torch::Tensor& block_ids,
    int64_t num_blocks)
{
    return expandBlockMeans(computeBlockMeans(values, block_ids, num_blocks), block_ids);
}

std::vector<float> decodeRestPayload(
    const EncodedRestPayload& encoded,
    const torch::Tensor& sh_levels,
    std::int64_t num_points,
    int max_sh_degree)
{
    std::vector<float> payload;
    payload.reserve(encoded.payload_values);
    if (encoded.blocks.empty() || encoded.payload_values == 0)
        return payload;

    auto levels_cpu = sh_levels.detach().contiguous().to(torch::kCPU, torch::kInt32);
    const std::int32_t* levels_ptr = levels_cpu.data_ptr<std::int32_t>();

    std::size_t residual_offset = 0;
    std::size_t base_offset = 0;
    std::int64_t point_offset = 0;
    for (const auto& block : encoded.blocks) {
        const int sh_level = std::clamp(static_cast<int>(block.sh_level), 0, max_sh_degree);
        const std::size_t payload_dims = restPayloadValuesForLevel(sh_level);
        const std::size_t block_points = static_cast<std::size_t>(block.point_count);
        if (point_offset + static_cast<std::int64_t>(block_points) > num_points)
            throw std::runtime_error("Decoded locality block layout exceeds point count.");

        for (std::size_t local_idx = 0; local_idx < block_points; ++local_idx) {
            const int point_level = std::clamp(static_cast<int>(levels_ptr[point_offset + static_cast<std::int64_t>(local_idx)]), 0, max_sh_degree);
            if (point_level != sh_level)
                throw std::runtime_error("Decoded locality block SH levels do not match stored point layout.");
        }

        if (payload_dims == 0) {
            point_offset += static_cast<std::int64_t>(block_points);
            continue;
        }

        if (base_offset + payload_dims > encoded.base_values.size() || base_offset + payload_dims > encoded.scale_values.size())
            throw std::runtime_error("Locality codec base/scale buffers are truncated.");

        const c10::Half* base_ptr = encoded.base_values.data() + static_cast<std::ptrdiff_t>(base_offset);
        const c10::Half* scale_ptr = encoded.scale_values.data() + static_cast<std::ptrdiff_t>(base_offset);
        const std::size_t residual_values = block_points * payload_dims;

        if (block.residual_bits == 4) {
            const std::size_t byte_count = (residual_values + 1) / 2;
            if (residual_offset + byte_count > encoded.residual_bytes.size())
                throw std::runtime_error("Locality codec int4 residual buffer is truncated.");
            for (std::size_t value_idx = 0; value_idx < residual_values; ++value_idx) {
                const std::uint8_t packed = encoded.residual_bytes[residual_offset + value_idx / 2];
                const int q = unpackInt4(packed, (value_idx & 1u) != 0u);
                const std::size_t dim = value_idx % payload_dims;
                payload.push_back(static_cast<float>(base_ptr[dim]) + static_cast<float>(q) * static_cast<float>(scale_ptr[dim]));
            }
            residual_offset += byte_count;
        }
        else if (block.residual_bits == 8) {
            if (residual_offset + residual_values > encoded.residual_bytes.size())
                throw std::runtime_error("Locality codec int8 residual buffer is truncated.");
            for (std::size_t value_idx = 0; value_idx < residual_values; ++value_idx) {
                const std::int8_t q = static_cast<std::int8_t>(encoded.residual_bytes[residual_offset + value_idx]);
                const std::size_t dim = value_idx % payload_dims;
                payload.push_back(static_cast<float>(base_ptr[dim]) + static_cast<float>(q) * static_cast<float>(scale_ptr[dim]));
            }
            residual_offset += residual_values;
        }
        else {
            throw std::runtime_error("Unsupported locality codec residual bitwidth.");
        }

        base_offset += payload_dims;
        point_offset += static_cast<std::int64_t>(block_points);
    }

    if (point_offset != num_points)
        throw std::runtime_error("Decoded locality block layout does not cover all points.");
    if (base_offset != encoded.base_values.size() || base_offset != encoded.scale_values.size())
        throw std::runtime_error("Decoded locality base/scale buffer size mismatch.");
    if (residual_offset != encoded.residual_bytes.size())
        throw std::runtime_error("Decoded locality residual buffer size mismatch.");
    if (payload.size() != encoded.payload_values)
        throw std::runtime_error("Decoded locality payload size mismatch.");

    return payload;
}

} // namespace locality_codec
