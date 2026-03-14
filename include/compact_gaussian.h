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

struct CompactExportOptions
{
    bool enable_export_prune = true;
    float prune_min_opacity = 0.008f;
    float prune_big_point_min_opacity = 0.025f;
    float prune_max_scaling_ratio = 0.15f;

    bool enable_sh_bandwidth = true;
    float sh_energy_keep_ratio = 0.995f;
    float sh_min_opacity = 0.01f;

    bool sort_by_morton = true;

    int xyz_quant_bits = 16;
    int attribute_quant_bits = 8;
    int rotation_quant_bits = 16;
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
