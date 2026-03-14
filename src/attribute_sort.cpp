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

#include "include/attribute_sort.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace
{

inline std::uint64_t splitBy3(std::uint32_t value)
{
    std::uint64_t expanded = value & 0x1fffff;
    expanded = (expanded | (expanded << 32)) & 0x1f00000000ffff;
    expanded = (expanded | (expanded << 16)) & 0x1f0000ff0000ff;
    expanded = (expanded | (expanded << 8)) & 0x100f00f00f00f00f;
    expanded = (expanded | (expanded << 4)) & 0x10c30c30c30c30c3;
    expanded = (expanded | (expanded << 2)) & 0x1249249249249249;
    return expanded;
}

inline std::uint64_t mortonEncode(std::uint32_t x, std::uint32_t y, std::uint32_t z)
{
    return splitBy3(x) | (splitBy3(y) << 1) | (splitBy3(z) << 2);
}

}

namespace attribute_sort
{

torch::Tensor mortonOrder(const torch::Tensor& xyz)
{
    if (!xyz.defined() || xyz.numel() == 0 || xyz.size(0) <= 1) {
        auto options = torch::TensorOptions().dtype(torch::kLong);
        if (xyz.defined())
            options = options.device(xyz.device());
        return torch::arange(xyz.defined() ? xyz.size(0) : 0, options);
    }

    auto xyz_cpu = xyz.detach().contiguous().to(torch::kCPU);
    const auto num_points = xyz_cpu.size(0);
    const float* xyz_ptr = xyz_cpu.data_ptr<float>();

    std::array<float, 3> mins = {
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    std::array<float, 3> maxs = {
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()};

    for (int64_t point_idx = 0; point_idx < num_points; ++point_idx) {
        for (int axis = 0; axis < 3; ++axis) {
            const float value = xyz_ptr[point_idx * 3 + axis];
            mins[axis] = std::min(mins[axis], value);
            maxs[axis] = std::max(maxs[axis], value);
        }
    }

    std::vector<std::pair<std::uint64_t, int64_t>> morton_pairs;
    morton_pairs.reserve(num_points);
    constexpr float quant_scale = static_cast<float>((1u << 21) - 1);
    for (int64_t point_idx = 0; point_idx < num_points; ++point_idx) {
        std::uint32_t coord[3] = {0, 0, 0};
        for (int axis = 0; axis < 3; ++axis) {
            const float range = maxs[axis] - mins[axis];
            float normalized = 0.0f;
            if (range > 1e-8f)
                normalized = (xyz_ptr[point_idx * 3 + axis] - mins[axis]) / range;
            normalized = std::clamp(normalized, 0.0f, 1.0f);
            coord[axis] = static_cast<std::uint32_t>(std::llround(normalized * quant_scale));
        }
        morton_pairs.emplace_back(mortonEncode(coord[0], coord[1], coord[2]), point_idx);
    }

    std::sort(
        morton_pairs.begin(),
        morton_pairs.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    std::vector<int64_t> order(num_points);
    for (int64_t point_idx = 0; point_idx < num_points; ++point_idx)
        order[point_idx] = morton_pairs[point_idx].second;

    auto order_tensor = torch::from_blob(
        order.data(),
        {num_points},
        torch::TensorOptions().dtype(torch::kLong)).clone();
    return order_tensor.to(xyz.device());
}

}
