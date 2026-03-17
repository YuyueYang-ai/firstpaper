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

#include "include/sh_bandwidth.h"

#include <algorithm>

namespace
{

inline int degreeOffset(int degree)
{
    return degree * degree - 1;
}

inline int degreeWidth(int degree)
{
    return 2 * degree + 1;
}

}

namespace sh_bandwidth
{

torch::Tensor estimateLevels(
    const torch::Tensor& features_rest,
    const torch::Tensor& opacity_activation,
    int max_sh_degree,
    float energy_keep_ratio,
    float min_opacity,
    int min_level)
{
    if (!features_rest.defined() || features_rest.numel() == 0 || max_sh_degree <= 0) {
        auto num_points = features_rest.defined() ? features_rest.size(0) : opacity_activation.size(0);
        return torch::zeros({num_points}, torch::TensorOptions().dtype(torch::kInt32).device(opacity_activation.device()));
    }

    min_level = std::clamp(min_level, 0, max_sh_degree);

    auto total_energy = torch::zeros(
        {features_rest.size(0)},
        torch::TensorOptions().dtype(torch::kFloat32).device(features_rest.device()));

    std::vector<torch::Tensor> degree_energy(max_sh_degree + 1);
    for (int degree = 1; degree <= max_sh_degree; ++degree) {
        const int start = degreeOffset(degree);
        const int width = degreeWidth(degree);
        degree_energy[degree] = features_rest
                                    .index({torch::indexing::Slice(),
                                            torch::indexing::Slice(start, start + width),
                                            torch::indexing::Slice()})
                                    .pow(2.0f)
                                    .sum({1, 2});
        total_energy += degree_energy[degree];
    }

    auto levels = torch::full(
        {features_rest.size(0)},
        max_sh_degree,
        torch::TensorOptions().dtype(torch::kInt32).device(features_rest.device()));
    auto cumulative_energy = torch::zeros_like(total_energy);
    auto unresolved = torch::ones_like(total_energy, torch::TensorOptions().dtype(torch::kBool));
    auto denominator = torch::clamp_min(total_energy, 1e-8f);

    for (int degree = 1; degree <= max_sh_degree; ++degree) {
        cumulative_energy += degree_energy[degree];
        auto reached = cumulative_energy / denominator >= energy_keep_ratio;
        auto assign = torch::logical_and(unresolved, reached);
        levels.index_put_({assign}, degree);
        unresolved = torch::logical_and(unresolved, torch::logical_not(reached));
    }

    auto low_energy = total_energy < 1e-8f;
    levels.index_put_({low_energy}, 0);

    if (opacity_activation.defined() && opacity_activation.numel() > 0) {
        auto low_opacity = opacity_activation.squeeze(-1) < min_opacity;
        levels.index_put_({low_opacity}, 0);
    }

    if (min_level > 0) {
        auto keep_floor_mask = levels > 0;
        if (keep_floor_mask.any().item<bool>()) {
            levels.index_put_(
                {keep_floor_mask},
                torch::clamp_min(levels.index({keep_floor_mask}), min_level));
        }
    }

    return levels;
}

torch::Tensor applyLevelsToFeaturesRest(
    const torch::Tensor& features_rest,
    const torch::Tensor& sh_levels,
    int max_sh_degree)
{
    if (!features_rest.defined() || features_rest.numel() == 0 || max_sh_degree <= 0
        || !sh_levels.defined() || sh_levels.numel() == 0) {
        return features_rest.clone();
    }

    auto masked_rest = features_rest.clone();
    auto levels = sh_levels.to(masked_rest.device()).to(torch::kInt32);
    for (int degree = 1; degree <= max_sh_degree; ++degree) {
        const int start = degreeOffset(degree);
        const int width = degreeWidth(degree);
        auto keep = (levels >= degree).to(masked_rest.dtype()).view({-1, 1, 1});
        auto degree_slice = masked_rest.index(
            {torch::indexing::Slice(),
             torch::indexing::Slice(start, start + width),
             torch::indexing::Slice()});
        masked_rest.index_put_(
            {torch::indexing::Slice(),
             torch::indexing::Slice(start, start + width),
             torch::indexing::Slice()},
            degree_slice * keep);
    }

    return masked_rest;
}

}
