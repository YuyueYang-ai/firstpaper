/**
 * This file is part of OmniGS
 */

#include "include/phase2_hybrid_selector.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <c10/util/Half.h>

#include "include/locality_codec.h"
#include "include/sh_bandwidth.h"

namespace
{

float quantileSorted(const std::vector<float>& sorted_values, float q)
{
    if (sorted_values.empty())
        return 0.0f;
    const float clamped_q = std::clamp(q, 0.0f, 1.0f);
    const float idx_f = clamped_q * static_cast<float>(sorted_values.size() - 1);
    const auto idx0 = static_cast<std::size_t>(std::floor(idx_f));
    const auto idx1 = static_cast<std::size_t>(std::ceil(idx_f));
    if (idx0 == idx1)
        return sorted_values[idx0];
    const float t = idx_f - static_cast<float>(idx0);
    return sorted_values[idx0] * (1.0f - t) + sorted_values[idx1] * t;
}

std::vector<float> robustNormalize(const std::vector<float>& values)
{
    if (values.empty())
        return {};

    std::vector<float> sorted_values(values.begin(), values.end());
    std::sort(sorted_values.begin(), sorted_values.end());
    const float p10 = quantileSorted(sorted_values, 0.10f);
    const float p90 = quantileSorted(sorted_values, 0.90f);
    const float denom = std::max(p90 - p10, 1e-8f);

    std::vector<float> normalized(values.size(), 0.0f);
    for (std::size_t idx = 0; idx < values.size(); ++idx)
        normalized[idx] = std::clamp((values[idx] - p10) / denom, 0.0f, 1.0f);
    return normalized;
}

struct ResidualQuantEstimate
{
    std::uint8_t residual_bits = 8;
    double relative_mse = 0.0;
    float explicit_bytes = 0.0f;
    float explicit_bpp = 0.0f;
};

ResidualQuantEstimate estimateExplicitStorage(
    const float* target_ptr,
    std::int64_t point_offset,
    std::size_t point_count,
    std::size_t payload_dims,
    std::int64_t dims_per_point,
    float int4_rel_mse_threshold)
{
    ResidualQuantEstimate estimate;
    if (point_count == 0) {
        estimate.explicit_bytes = 0.0f;
        estimate.explicit_bpp = 0.0f;
        return estimate;
    }

    constexpr float kBlockInfoBytes = 4.0f;
    if (payload_dims == 0) {
        estimate.residual_bits = 8;
        estimate.relative_mse = 0.0;
        estimate.explicit_bytes = kBlockInfoBytes;
        estimate.explicit_bpp = estimate.explicit_bytes / static_cast<float>(point_count);
        return estimate;
    }

    std::vector<float> base(payload_dims, 0.0f);
    for (std::size_t local_idx = 0; local_idx < point_count; ++local_idx) {
        const float* point_ptr = target_ptr + ((point_offset + static_cast<std::int64_t>(local_idx)) * dims_per_point);
        for (std::size_t dim = 0; dim < payload_dims; ++dim)
            base[dim] += point_ptr[dim];
    }
    for (float& value : base)
        value /= static_cast<float>(point_count);

    std::vector<float> residuals(point_count * payload_dims, 0.0f);
    for (std::size_t local_idx = 0; local_idx < point_count; ++local_idx) {
        const float* point_ptr = target_ptr + ((point_offset + static_cast<std::int64_t>(local_idx)) * dims_per_point);
        for (std::size_t dim = 0; dim < payload_dims; ++dim)
            residuals[local_idx * payload_dims + dim] = point_ptr[dim] - base[dim];
    }

    auto relativeMseForBits = [&](float qmax) {
        std::vector<float> scales(payload_dims, 0.0f);
        for (std::size_t dim = 0; dim < payload_dims; ++dim) {
            float max_abs = 0.0f;
            for (std::size_t local_idx = 0; local_idx < point_count; ++local_idx)
                max_abs = std::max(max_abs, std::abs(residuals[local_idx * payload_dims + dim]));
            scales[dim] = max_abs > 1e-8f ? (max_abs / qmax) : 0.0f;
        }

        double sum_sq = 0.0;
        double err_sq = 0.0;
        for (std::size_t idx = 0; idx < residuals.size(); ++idx) {
            const std::size_t dim = idx % payload_dims;
            const float scale = scales[dim];
            int q = 0;
            if (scale > 0.0f)
                q = static_cast<int>(std::llround(static_cast<double>(residuals[idx]) / static_cast<double>(scale)));
            if (qmax == 7.0f)
                q = std::clamp(q, -8, 7);
            else
                q = std::clamp(q, -127, 127);
            const float reconstructed = static_cast<float>(q) * scale;
            const double diff = static_cast<double>(residuals[idx] - reconstructed);
            err_sq += diff * diff;
            sum_sq += static_cast<double>(residuals[idx]) * static_cast<double>(residuals[idx]);
        }
        return err_sq / std::max(1e-12, sum_sq);
    };

    const double int4_rel_mse = relativeMseForBits(7.0f);
    const bool use_int4 = int4_rel_mse <= static_cast<double>(int4_rel_mse_threshold);
    estimate.residual_bits = use_int4 ? 4 : 8;
    estimate.relative_mse = use_int4 ? int4_rel_mse : relativeMseForBits(127.0f);

    const std::size_t residual_values = point_count * payload_dims;
    const float residual_bytes = use_int4
        ? static_cast<float>((residual_values + 1) / 2)
        : static_cast<float>(residual_values);
    const float base_scale_bytes = static_cast<float>(payload_dims * sizeof(c10::Half) * 2);
    estimate.explicit_bytes = kBlockInfoBytes + base_scale_bytes + residual_bytes;
    estimate.explicit_bpp = estimate.explicit_bytes / static_cast<float>(point_count);
    return estimate;
}

template <typename T>
torch::Tensor tensorFromVector(
    const std::vector<T>& values,
    torch::ScalarType dtype,
    torch::Device device)
{
    auto tensor = torch::empty(
        {static_cast<int64_t>(values.size())},
        torch::TensorOptions().dtype(dtype).device(torch::kCPU));
    if (!values.empty())
        std::memcpy(tensor.data_ptr<T>(), values.data(), values.size() * sizeof(T));
    return tensor.to(device);
}

} // namespace

namespace phase2_hybrid_selector
{

HybridSelectionResult selectHardBlocks(
    const torch::Tensor& features_rest_target,
    const torch::Tensor& features_rest_base,
    const torch::Tensor& sh_levels,
    const torch::Tensor& block_ids,
    int64_t num_blocks,
    int max_sh_degree,
    const Phase2ResidualFieldOptions::HybridSelectorOptions& options)
{
    HybridSelectionResult result;
    result.enabled = options.enable;
    if (!options.enable)
        return result;

    if (!features_rest_target.defined() || !features_rest_base.defined())
        throw std::runtime_error("Hybrid selector expects defined target/base tensors.");
    if (!sh_levels.defined() || !block_ids.defined())
        throw std::runtime_error("Hybrid selector expects defined sh_levels and block_ids.");
    if (features_rest_target.size(0) != features_rest_base.size(0)
        || features_rest_target.size(0) != sh_levels.size(0)
        || features_rest_target.size(0) != block_ids.size(0))
        throw std::runtime_error("Hybrid selector: tensor length mismatch.");
    if (num_blocks <= 0)
        throw std::runtime_error("Hybrid selector: expected positive block count.");

    auto device = features_rest_target.device();
    auto target_cpu = features_rest_target.detach().contiguous().to(torch::kCPU, torch::kFloat32);
    auto base_cpu = features_rest_base.detach().contiguous().to(torch::kCPU, torch::kFloat32);
    auto levels_cpu = sh_levels.detach().contiguous().to(torch::kCPU, torch::kInt32);
    auto block_ids_cpu = block_ids.detach().contiguous().to(torch::kCPU, torch::kLong);

    auto residual = target_cpu - base_cpu;
    auto mask = sh_bandwidth::applyLevelsToFeaturesRest(
        torch::ones_like(target_cpu),
        levels_cpu,
        max_sh_degree).to(torch::kFloat32);
    auto residual_masked = residual * mask;

    auto residual_flat = residual_masked.view({residual_masked.size(0), -1});
    auto mask_flat = mask.view({mask.size(0), -1});
    auto point_active_counts = torch::clamp_min(mask_flat.sum(1), 1.0f);
    auto point_mse = residual_flat.pow(2).sum(1) / point_active_counts;
    auto point_absmax = std::get<0>(residual_flat.abs().max(1));

    std::vector<float> block_mean_mse(static_cast<std::size_t>(num_blocks), 0.0f);
    std::vector<float> block_max_abs(static_cast<std::size_t>(num_blocks), 0.0f);
    std::vector<float> block_explicit_bytes(static_cast<std::size_t>(num_blocks), 0.0f);
    std::vector<float> block_explicit_bpp(static_cast<std::size_t>(num_blocks), 0.0f);
    std::vector<std::int32_t> block_point_counts(static_cast<std::size_t>(num_blocks), 0);
    std::vector<std::int32_t> block_levels(static_cast<std::size_t>(num_blocks), 0);

    const float* point_mse_ptr = point_mse.data_ptr<float>();
    const float* point_absmax_ptr = point_absmax.data_ptr<float>();
    const float* target_ptr = target_cpu.data_ptr<float>();
    const std::int32_t* level_ptr = levels_cpu.data_ptr<std::int32_t>();
    const int64_t* block_ids_ptr = block_ids_cpu.data_ptr<int64_t>();
    const int64_t num_points = features_rest_target.size(0);
    const std::int64_t dims_per_point = target_cpu.size(1) * target_cpu.size(2);

    for (int64_t point_idx = 0; point_idx < num_points; ++point_idx) {
        const int64_t block_idx = block_ids_ptr[point_idx];
        if (block_idx < 0 || block_idx >= num_blocks)
            throw std::runtime_error("Hybrid selector: block id out of range.");
        auto& mse_sum = block_mean_mse[static_cast<std::size_t>(block_idx)];
        auto& abs_max = block_max_abs[static_cast<std::size_t>(block_idx)];
        auto& count = block_point_counts[static_cast<std::size_t>(block_idx)];
        auto& level = block_levels[static_cast<std::size_t>(block_idx)];
        mse_sum += point_mse_ptr[point_idx];
        abs_max = std::max(abs_max, point_absmax_ptr[point_idx]);
        count += 1;
        level = std::max(level, static_cast<std::int32_t>(std::clamp(level_ptr[point_idx], 0, max_sh_degree)));
    }

    for (int64_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const auto count = std::max<std::int32_t>(1, block_point_counts[static_cast<std::size_t>(block_idx)]);
        block_mean_mse[static_cast<std::size_t>(block_idx)] /= static_cast<float>(count);
    }

    const auto mse_norm = robustNormalize(block_mean_mse);
    const auto abs_norm = robustNormalize(block_max_abs);
    std::int64_t point_offset = 0;
    for (int64_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const auto block_points = static_cast<std::size_t>(std::max<std::int32_t>(1, block_point_counts[static_cast<std::size_t>(block_idx)]));
        const int sh_level = std::clamp(block_levels[static_cast<std::size_t>(block_idx)], 0, max_sh_degree);
        const std::size_t payload_dims = locality_codec::restPayloadValuesForLevel(sh_level);
        const auto estimate = estimateExplicitStorage(
            target_ptr,
            point_offset,
            block_points,
            payload_dims,
            dims_per_point,
            options.explicit_cost_int4_rel_mse_threshold);
        block_explicit_bytes[static_cast<std::size_t>(block_idx)] = estimate.explicit_bytes;
        block_explicit_bpp[static_cast<std::size_t>(block_idx)] = estimate.explicit_bpp;
        point_offset += static_cast<std::int64_t>(block_points);
    }
    const auto cost_norm = robustNormalize(block_explicit_bpp);

    std::vector<float> block_scores(static_cast<std::size_t>(num_blocks), 0.0f);
    for (int64_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const float level_norm = max_sh_degree > 0
            ? static_cast<float>(block_levels[static_cast<std::size_t>(block_idx)]) / static_cast<float>(max_sh_degree)
            : 0.0f;
        block_scores[static_cast<std::size_t>(block_idx)] =
            options.alpha * mse_norm[static_cast<std::size_t>(block_idx)]
            + options.beta * abs_norm[static_cast<std::size_t>(block_idx)]
            + options.gamma * cost_norm[static_cast<std::size_t>(block_idx)]
            + options.delta * level_norm;
    }

    std::vector<int64_t> sorted_block_ids(static_cast<std::size_t>(num_blocks), 0);
    std::iota(sorted_block_ids.begin(), sorted_block_ids.end(), int64_t{0});
    std::stable_sort(
        sorted_block_ids.begin(),
        sorted_block_ids.end(),
        [&](int64_t lhs, int64_t rhs) {
            return block_scores[static_cast<std::size_t>(lhs)] > block_scores[static_cast<std::size_t>(rhs)];
        });

    const int64_t target_hard_points = std::clamp<int64_t>(
        static_cast<int64_t>(std::ceil(std::clamp(options.hard_point_ratio, 0.0f, 1.0f) * static_cast<float>(num_points))),
        0,
        num_points);
    const int max_hard_blocks = options.max_hard_blocks > 0
        ? std::max(options.max_hard_blocks, options.min_hard_blocks)
        : -1;

    std::vector<std::int32_t> hard_block_flags(static_cast<std::size_t>(num_blocks), 0);
    std::vector<int64_t> hard_block_ids;
    hard_block_ids.reserve(static_cast<std::size_t>(num_blocks));
    int64_t hard_points = 0;

    for (const auto block_idx : sorted_block_ids) {
        const bool need_more_points = hard_points < target_hard_points;
        const bool need_more_blocks = static_cast<int>(hard_block_ids.size()) < std::max(0, options.min_hard_blocks);
        if (!need_more_points && !need_more_blocks)
            break;
        if (max_hard_blocks > 0 && static_cast<int>(hard_block_ids.size()) >= max_hard_blocks)
            break;
        hard_block_flags[static_cast<std::size_t>(block_idx)] = 1;
        hard_block_ids.push_back(block_idx);
        hard_points += static_cast<int64_t>(block_point_counts[static_cast<std::size_t>(block_idx)]);
    }

    std::vector<std::int32_t> hard_point_flags(static_cast<std::size_t>(num_points), 0);
    for (int64_t point_idx = 0; point_idx < num_points; ++point_idx)
        hard_point_flags[static_cast<std::size_t>(point_idx)] =
            hard_block_flags[static_cast<std::size_t>(block_ids_ptr[point_idx])];

    auto meanForFlag = [&](const std::vector<float>& values, const std::vector<std::int32_t>& flags, int flag_value) -> float {
        double accum = 0.0;
        int64_t count = 0;
        for (std::size_t idx = 0; idx < values.size(); ++idx) {
            if (flags[idx] != flag_value)
                continue;
            accum += values[idx];
            ++count;
        }
        return count > 0 ? static_cast<float>(accum / static_cast<double>(count)) : 0.0f;
    };
    std::vector<float> block_levels_float(static_cast<std::size_t>(num_blocks), 0.0f);
    for (std::size_t idx = 0; idx < block_levels.size(); ++idx)
        block_levels_float[idx] = static_cast<float>(block_levels[idx]);

    result.block_scores = tensorFromVector<float>(block_scores, torch::kFloat32, device);
    result.block_mean_mse = tensorFromVector<float>(block_mean_mse, torch::kFloat32, device);
    result.block_max_abs = tensorFromVector<float>(block_max_abs, torch::kFloat32, device);
    result.block_point_counts = tensorFromVector<std::int32_t>(block_point_counts, torch::kInt32, device);
    result.block_levels = tensorFromVector<std::int32_t>(block_levels, torch::kInt32, device);
    result.block_explicit_bytes = tensorFromVector<float>(block_explicit_bytes, torch::kFloat32, device);
    result.block_explicit_bpp = tensorFromVector<float>(block_explicit_bpp, torch::kFloat32, device);
    result.hard_block_flags = tensorFromVector<std::int32_t>(hard_block_flags, torch::kInt32, device);
    result.hard_point_flags = tensorFromVector<std::int32_t>(hard_point_flags, torch::kInt32, device);
    result.hard_block_ids = tensorFromVector<int64_t>(hard_block_ids, torch::kLong, device);
    result.num_hard_blocks = static_cast<int64_t>(hard_block_ids.size());
    result.num_hard_points = hard_points;
    result.realized_hard_block_ratio = num_blocks > 0
        ? static_cast<float>(result.num_hard_blocks) / static_cast<float>(num_blocks)
        : 0.0f;
    result.realized_hard_point_ratio = num_points > 0
        ? static_cast<float>(result.num_hard_points) / static_cast<float>(num_points)
        : 0.0f;
    result.mean_score_hard = meanForFlag(block_scores, hard_block_flags, 1);
    result.mean_score_easy = meanForFlag(block_scores, hard_block_flags, 0);
    result.mean_block_mse_hard = meanForFlag(block_mean_mse, hard_block_flags, 1);
    result.mean_block_mse_easy = meanForFlag(block_mean_mse, hard_block_flags, 0);
    result.mean_sh_level_hard = meanForFlag(block_levels_float, hard_block_flags, 1);
    result.mean_sh_level_easy = meanForFlag(block_levels_float, hard_block_flags, 0);
    result.mean_block_explicit_bpp_hard = meanForFlag(block_explicit_bpp, hard_block_flags, 1);
    result.mean_block_explicit_bpp_easy = meanForFlag(block_explicit_bpp, hard_block_flags, 0);
    return result;
}

} // namespace phase2_hybrid_selector
