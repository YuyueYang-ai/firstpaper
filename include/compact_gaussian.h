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

#include <torch/torch.h>

enum class CompressionMode
{
    BASELINE = 0,
    COMPACT = 1,
    COMPACT_PHASE2 = 2
};

inline const char* compressionModeName(CompressionMode mode)
{
    switch (mode) {
    case CompressionMode::BASELINE:
        return "baseline";
    case CompressionMode::COMPACT:
        return "compact";
    case CompressionMode::COMPACT_PHASE2:
        return "compact_phase2";
    }
    return "baseline";
}

struct CompactExportOptions
{
    bool enable_export_prune = true;
    float prune_min_opacity = 0.008f;
    float prune_big_point_min_opacity = 0.02f;
    float prune_max_scaling_ratio = 0.15f;

    bool enable_sh_bandwidth = true;
    float sh_energy_keep_ratio = 0.995f;
    float sh_min_opacity = 0.01f;
    int sh_min_level = 0;

    bool sort_by_morton = true;
    bool f_rest_blockwise_quant = false;
    int f_rest_block_size = 128;
    bool f_rest_locality_codec = false;
    int f_rest_locality_high_sh_block_size = 64;
    int f_rest_locality_low_sh_block_size = 128;
    float f_rest_locality_int4_rel_mse_threshold = 0.02f;

    int xyz_quant_bits = 16;
    int attribute_quant_bits = 8;
    int rotation_quant_bits = 16;
};

struct Phase2ResidualFieldOptions
{
    bool enable = false;
    bool save_frozen_snapshot = false;
    int freeze_topology_iter = -1;
    bool sort_by_morton = true;
    bool normalize_xyz = true;
    bool mask_features_rest_by_sh_level = true;
    bool use_locality_base = true;
    int locality_high_sh_block_size = 64;
    int locality_low_sh_block_size = 128;
};

struct Phase2ResidualFieldTrainOptions
{
    int num_fourier_frequencies = 6;
    int hidden_dim = 128;
    int num_hidden_layers = 3;
    int batch_size = 8192;
    int max_steps = 4000;
    int log_interval = 200;
    int eval_interval = 500;
    float learning_rate = 1e-3f;
    float weight_decay = 1e-6f;
    bool include_features_dc = true;
    bool include_opacity = true;
    bool include_scaling = true;
    bool include_rotation = true;
    int block_embedding_dim = 8;
    bool save_decoded_compact = true;
    bool save_phase2_compact = true;
    int decoded_xyz_quant_bits = 16;
    int decoded_attribute_quant_bits = 16;
    int decoded_rotation_quant_bits = 16;
};

struct DecodedGaussianTensors
{
    int max_sh_degree = 0;
    int active_sh_degree = 0;

    torch::Tensor xyz;
    torch::Tensor features_dc;
    torch::Tensor features_rest;
    torch::Tensor opacity;
    torch::Tensor scaling;
    torch::Tensor rotation;
    torch::Tensor sh_levels;

    inline bool empty() const
    {
        return !xyz.defined() || xyz.numel() == 0;
    }
};
