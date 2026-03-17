/**
 * This file is part of OmniGS
 */

#include "include/hashgrid_field.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

HashGridEncoderImpl::HashGridEncoderImpl(const HashGridEncoderOptions& options)
{
    num_levels_ = std::max(1, options.num_levels);
    features_per_level_ = std::max(1, options.features_per_level);
    table_size_ = 1 << std::clamp(options.log2_hashmap_size, 8, 22);
    resolutions_.resize(static_cast<std::size_t>(num_levels_));
    tables_.resize(static_cast<std::size_t>(num_levels_));

    for (int level_idx = 0; level_idx < num_levels_; ++level_idx) {
        const float resolution_f = static_cast<float>(options.base_resolution)
            * std::pow(options.per_level_scale, static_cast<float>(level_idx));
        const int resolution = std::max(2, static_cast<int>(std::llround(resolution_f)));
        resolutions_[static_cast<std::size_t>(level_idx)] = resolution;
        auto table = register_parameter(
            "hash_table_" + std::to_string(level_idx),
            torch::randn({table_size_, features_per_level_}, torch::TensorOptions().dtype(torch::kFloat32)) * 1e-4f);
        tables_[static_cast<std::size_t>(level_idx)] = table;
    }
}

torch::Tensor HashGridEncoderImpl::hashIndices(const torch::Tensor& coords) const
{
    if (coords.dim() != 2 || coords.size(1) != 3)
        throw std::runtime_error("HashGridEncoder expects integer coords with shape [N, 3].");

    auto coords_i64 = coords.to(torch::kInt64);
    auto x = coords_i64.select(1, 0);
    auto y = coords_i64.select(1, 1);
    auto z = coords_i64.select(1, 2);
    auto hashed = x * c10::Scalar(static_cast<int64_t>(73856093))
        + y * c10::Scalar(static_cast<int64_t>(19349663))
        + z * c10::Scalar(static_cast<int64_t>(83492791));
    return torch::remainder(hashed, table_size_).to(torch::kLong);
}

torch::Tensor HashGridEncoderImpl::forward(const torch::Tensor& xyz_normalized)
{
    auto xyz = ((xyz_normalized.to(torch::kFloat32) + 1.0f) * 0.5f).clamp(0.0f, 1.0f);
    std::vector<torch::Tensor> level_features;
    level_features.reserve(static_cast<std::size_t>(num_levels_));

    for (int level_idx = 0; level_idx < num_levels_; ++level_idx) {
        const int resolution = resolutions_[static_cast<std::size_t>(level_idx)];
        auto scaled = xyz * static_cast<float>(resolution - 1);
        auto base = torch::floor(scaled).to(torch::kInt64);
        auto frac = scaled - base.to(torch::kFloat32);

        auto accum = torch::zeros(
            {xyz.size(0), features_per_level_},
            torch::TensorOptions().dtype(torch::kFloat32).device(xyz.device()));

        for (int corner = 0; corner < 8; ++corner) {
            const int ox = corner & 1;
            const int oy = (corner >> 1) & 1;
            const int oz = (corner >> 2) & 1;
            auto offset = torch::tensor(
                {ox, oy, oz},
                torch::TensorOptions().dtype(torch::kInt64).device(xyz.device()));
            auto coords = torch::clamp(base + offset, 0, resolution - 1);
            auto idx = hashIndices(coords);

            auto wx = ox ? frac.select(1, 0) : (1.0f - frac.select(1, 0));
            auto wy = oy ? frac.select(1, 1) : (1.0f - frac.select(1, 1));
            auto wz = oz ? frac.select(1, 2) : (1.0f - frac.select(1, 2));
            auto weight = (wx * wy * wz).unsqueeze(1);

            auto table = tables_[static_cast<std::size_t>(level_idx)];
            auto feats = torch::index_select(table, 0, idx);
            accum = accum + feats * weight;
        }

        level_features.push_back(accum);
    }

    return torch::cat(level_features, 1);
}
