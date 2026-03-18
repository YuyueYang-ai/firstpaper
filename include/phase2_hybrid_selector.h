/**
 * This file is part of OmniGS
 */

#pragma once

#include <torch/torch.h>

#include "compact_gaussian.h"

namespace phase2_hybrid_selector
{

struct HybridSelectionResult
{
    bool enabled = false;
    torch::Tensor block_scores;
    torch::Tensor block_mean_mse;
    torch::Tensor block_max_abs;
    torch::Tensor block_point_counts;
    torch::Tensor block_levels;
    torch::Tensor block_explicit_bytes;
    torch::Tensor block_explicit_bpp;
    torch::Tensor hard_block_flags;
    torch::Tensor hard_point_flags;
    torch::Tensor hard_block_ids;
    int64_t num_hard_blocks = 0;
    int64_t num_hard_points = 0;
    float realized_hard_point_ratio = 0.0f;
    float realized_hard_block_ratio = 0.0f;
    float mean_score_hard = 0.0f;
    float mean_score_easy = 0.0f;
    float mean_block_mse_hard = 0.0f;
    float mean_block_mse_easy = 0.0f;
    float mean_sh_level_hard = 0.0f;
    float mean_sh_level_easy = 0.0f;
    float mean_block_explicit_bpp_hard = 0.0f;
    float mean_block_explicit_bpp_easy = 0.0f;
};

HybridSelectionResult selectHardBlocks(
    const torch::Tensor& features_rest_target,
    const torch::Tensor& features_rest_base,
    const torch::Tensor& sh_levels,
    const torch::Tensor& block_ids,
    int64_t num_blocks,
    int max_sh_degree,
    const Phase2ResidualFieldOptions::HybridSelectorOptions& options);

} // namespace phase2_hybrid_selector
