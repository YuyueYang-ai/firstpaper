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

#include <filesystem>

#include <torch/torch.h>

#include "compact_gaussian.h"
#include "hashgrid_field.h"

namespace phase2_residual_field
{

struct FrozenResidualFieldPackage
{
    int max_sh_degree = 0;
    int active_sh_degree = 0;
    float scene_extent = 0.0f;
    bool use_locality_base = false;
    int locality_high_sh_block_size = 64;
    int locality_low_sh_block_size = 128;

    torch::Tensor xyz;
    torch::Tensor xyz_normalized;
    torch::Tensor features_dc;
    torch::Tensor features_rest_base;
    torch::Tensor features_rest_target;
    torch::Tensor opacity_base;
    torch::Tensor opacity;
    torch::Tensor scaling_base;
    torch::Tensor scaling;
    torch::Tensor rotation_base;
    torch::Tensor rotation;
    torch::Tensor sh_levels;
    torch::Tensor block_ids;
    torch::Tensor block_scores;
    torch::Tensor block_mean_mse;
    torch::Tensor block_max_abs;
    torch::Tensor block_point_counts;
    torch::Tensor block_levels;
    torch::Tensor hard_block_flags;
    torch::Tensor hard_point_flags;
    torch::Tensor hard_block_ids;
    torch::Tensor morton_order;
    torch::Tensor inverse_morton_order;
    int64_t num_blocks = 0;
    int64_t num_hard_blocks = 0;
    int64_t num_hard_points = 0;
};

struct Phase2ResidualFieldTrainResult
{
    float best_loss = 0.0f;
    float final_eval_loss = 0.0f;
    int trained_steps = 0;
    DecodedGaussianTensors decoded;
};

class Phase2ResidualFieldImpl : public torch::nn::Module
{
public:
    Phase2ResidualFieldImpl(
        int max_sh_degree,
        int64_t num_blocks,
        const Phase2ResidualFieldTrainOptions& options);

    torch::Tensor forward(
        const torch::Tensor& xyz_normalized,
        const torch::Tensor& sh_levels,
        const torch::Tensor& features_dc,
        const torch::Tensor& opacity,
        const torch::Tensor& scaling,
        const torch::Tensor& rotation,
        const torch::Tensor& block_ids);

    int outputDim() const { return output_dim_; }
    int restOutputDim() const { return rest_output_dim_; }
    bool predictsOpacity() const { return predict_opacity_; }
    bool predictsScaling() const { return predict_scaling_; }
    bool predictsRotation() const { return predict_rotation_; }

private:
    torch::Tensor encode(
        const torch::Tensor& xyz_normalized,
        const torch::Tensor& sh_levels,
        const torch::Tensor& features_dc,
        const torch::Tensor& opacity,
        const torch::Tensor& scaling,
        const torch::Tensor& rotation,
        const torch::Tensor& block_ids);

private:
    int max_sh_degree_ = 0;
    int64_t num_blocks_ = 0;
    int num_fourier_frequencies_ = 0;
    bool use_hashgrid_encoder_ = false;
    int output_dim_ = 0;
    int rest_output_dim_ = 0;
    bool include_features_dc_ = true;
    bool include_opacity_ = true;
    bool include_scaling_ = true;
    bool include_rotation_ = true;
    bool predict_opacity_ = true;
    bool predict_scaling_ = true;
    bool predict_rotation_ = true;
    int block_embedding_dim_ = 0;
    torch::nn::Embedding block_embedding_{nullptr};
    HashGridEncoder hashgrid_encoder_{nullptr};
    torch::nn::Sequential network_{nullptr};
};

TORCH_MODULE(Phase2ResidualField);

void saveFrozenPackage(
    const DecodedGaussianTensors& decoded,
    const std::filesystem::path& result_dir,
    float scene_extent,
    const Phase2ResidualFieldOptions& options);

FrozenResidualFieldPackage loadFrozenPackage(
    const std::filesystem::path& result_dir,
    torch::DeviceType device_type);

DecodedGaussianTensors loadPhase2Compact(
    const std::filesystem::path& result_dir,
    torch::DeviceType device_type);

Phase2ResidualFieldTrainResult trainResidualField(
    const FrozenResidualFieldPackage& frozen,
    const Phase2ResidualFieldTrainOptions& options,
    const std::filesystem::path& result_dir);

} // namespace phase2_residual_field
