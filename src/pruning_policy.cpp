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

#include "include/pruning_policy.h"

#include "include/gaussian_model.h"

namespace pruning_policy
{

torch::Tensor buildCompactExportMask(
    GaussianModel& model,
    float scene_extent,
    const CompactExportOptions& options)
{
    auto opacity = model.getOpacityActivation().squeeze(-1);
    auto prune_mask = opacity < options.prune_min_opacity;

    if (options.prune_max_scaling_ratio > 0.0f && scene_extent > 0.0f) {
        auto scales = model.getScalingActivation();
        auto big_points = std::get<0>(scales.max(/*dim=*/1)) > options.prune_max_scaling_ratio * scene_extent;
        auto low_contribution_big_points = torch::logical_and(
            big_points,
            opacity < options.prune_big_point_min_opacity);
        prune_mask = torch::logical_or(prune_mask, low_contribution_big_points);
    }

    return prune_mask;
}

}
