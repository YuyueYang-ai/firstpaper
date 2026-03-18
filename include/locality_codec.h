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

#pragma once

#include <cstdint>
#include <vector>

#include <c10/util/Half.h>
#include <torch/torch.h>

#include "compact_gaussian.h"

namespace locality_codec
{

struct RestBlockInfo
{
    std::uint8_t sh_level = 0;
    std::uint16_t point_count = 0;
    std::uint8_t residual_bits = 8;
};

struct EncodedRestPayload
{
    std::vector<RestBlockInfo> blocks;
    std::vector<c10::Half> base_values;
    std::vector<c10::Half> scale_values;
    std::vector<std::uint8_t> residual_bytes;
    std::vector<std::uint8_t> residual_bytes_int2;
    std::vector<std::uint8_t> residual_bytes_int4;
    std::vector<std::uint8_t> residual_bytes_int8;
    std::size_t payload_values = 0;
    std::size_t int2_block_count = 0;
    std::size_t int4_block_count = 0;
    std::size_t int8_block_count = 0;
};

EncodedRestPayload encodeRestPayload(
    const torch::Tensor& features_rest,
    const torch::Tensor& sh_levels,
    int max_sh_degree,
    const CompactExportOptions& options);

EncodedRestPayload encodeRestPayloadPreserveBlocks(
    const torch::Tensor& features_rest,
    const torch::Tensor& block_ids,
    const torch::Tensor& block_export_levels,
    int max_sh_degree,
    const CompactExportOptions& options);

torch::Tensor computeRestBlockBase(
    const torch::Tensor& features_rest,
    const torch::Tensor& sh_levels,
    int max_sh_degree,
    const CompactExportOptions& options);

torch::Tensor computeBlockMeans(
    const torch::Tensor& values,
    const torch::Tensor& block_ids,
    int64_t num_blocks);

torch::Tensor expandBlockMeans(
    const torch::Tensor& block_means,
    const torch::Tensor& block_ids);

torch::Tensor computeExpandedBlockMeans(
    const torch::Tensor& values,
    const torch::Tensor& block_ids,
    int64_t num_blocks);

torch::Tensor computeRestBlockBases(
    const torch::Tensor& features_rest,
    const torch::Tensor& sh_levels,
    int max_sh_degree,
    const CompactExportOptions& options);

torch::Tensor expandRestBlockBases(
    const torch::Tensor& block_bases,
    const torch::Tensor& sh_levels,
    int max_sh_degree,
    const CompactExportOptions& options);

torch::Tensor computeRestBlockIds(
    const torch::Tensor& sh_levels,
    int max_sh_degree,
    const CompactExportOptions& options);

std::vector<float> decodeRestPayload(
    const EncodedRestPayload& encoded,
    const torch::Tensor& sh_levels,
    std::int64_t num_points,
    int max_sh_degree);

std::size_t restPayloadValuesForLevel(int level);

} // namespace locality_codec
