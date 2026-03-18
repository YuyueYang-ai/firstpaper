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
    float f_rest_locality_int2_rel_mse_threshold = 0.0f;
    float f_rest_locality_int4_rel_mse_threshold = 0.02f;

    int xyz_quant_bits = 16;
    int attribute_quant_bits = 8;
    int rotation_quant_bits = 16;
};

struct Phase2ResidualFieldOptions
{
    struct HybridSelectorOptions
    {
        bool enable = false;
        float hard_point_ratio = 0.15f;
        float alpha = 0.45f;
        float beta = 0.10f;
        float gamma = 0.35f;
        float delta = 0.10f;
        float explicit_cost_int4_rel_mse_threshold = 0.02f;
        int min_hard_blocks = 1;
        int max_hard_blocks = -1;
        bool save_debug_tensors = true;
    };

    bool enable = false;
    bool save_frozen_snapshot = false;
    int freeze_topology_iter = -1;
    bool sort_by_morton = true;
    bool normalize_xyz = true;
    bool mask_features_rest_by_sh_level = true;
    bool use_locality_base = true;
    int locality_high_sh_block_size = 64;
    int locality_low_sh_block_size = 128;
    HybridSelectorOptions hybrid_selector;
};

struct Phase2ResidualFieldTrainOptions
{
    int num_fourier_frequencies = 6;
    bool use_hashgrid_encoder = true;
    int hashgrid_num_levels = 8;
    int hashgrid_features_per_level = 2;
    int hashgrid_log2_hashmap_size = 18;
    int hashgrid_base_resolution = 16;
    float hashgrid_per_level_scale = 1.5f;
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
    bool predict_opacity = true;
    bool predict_scaling = true;
    bool predict_rotation = true;
    int block_embedding_dim = 8;
    bool hybrid_hard_only = false;
    bool hybrid_override_rest_only = true;
    bool hybrid_easy_export_sh_drop = false;
    bool hybrid_easy_export_sh_preserve_blocks = true;
    float hybrid_easy_export_sh_energy_keep_ratio = 0.995f;
    float hybrid_easy_export_sh_min_opacity = 0.01f;
    int hybrid_easy_export_sh_min_level = 0;
    bool save_decoded_compact = true;
    bool save_phase2_compact = true;
    int decoded_xyz_quant_bits = 16;
    int decoded_attribute_quant_bits = 16;
    int decoded_rotation_quant_bits = 16;
    int phase2_compact_opacity_quant_bits = 8;
    int phase2_compact_scaling_quant_bits = 8;
    bool phase2_compact_pack_sh_levels = true;
    int phase2_compact_fdc_quant_bits = 8;
    int phase2_compact_easy_rest_base_quant_bits = 16;
    int phase2_compact_easy_rest_scale_quant_bits = 16;
    float phase2_compact_easy_rest_int2_rel_mse_threshold = 0.0f;
    bool phase2_compact_use_geometry_codec = true;
    int phase2_compact_geometry_quant_bits = 16;
    bool phase2_compact_store_field_fp16 = true;
    bool phase2_compact_easy_rest_zlib = true;
    int phase2_compact_easy_rest_zlib_level = 6;
    bool phase2_compact_quantized_tensor_zlib = true;
    int phase2_compact_quantized_tensor_zlib_level = 6;
    bool phase2_compact_geometry_zlib = true;
    int phase2_compact_geometry_zlib_level = 6;
    bool phase2_compact_field_zlib = true;
    int phase2_compact_field_zlib_level = 6;
    bool phase2_compact_use_xz = false;
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
