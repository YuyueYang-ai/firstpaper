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

#include <string>
#include <filesystem>

class GaussianModelParams
{
public:
    GaussianModelParams(
        std::filesystem::path source_path = "",
        std::filesystem::path model_path = "",
        std::filesystem::path exec_path = "",
        int sh_degree = 3,
        std::string images = "images",
        float resolution = -1.0f,
        bool white_background = false,
        std::string data_device = "cuda",
        bool eval = false);

public:
    int sh_degree_;
    std::filesystem::path source_path_;
    std::filesystem::path model_path_;
    std::string images_;
    float resolution_;
    bool white_background_;
    std::string data_device_;
    bool eval_;
};

class GaussianPipelineParams
{
public:
    GaussianPipelineParams(
        bool convert_SHs = false,
        bool compute_cov3D = false);

public:
    bool convert_SHs_;
    bool compute_cov3D_;
};

class GaussianOptimizationParams
{
public:
    GaussianOptimizationParams(
        int iterations = 30'000,
        float position_lr_init = 0.00016f,
        float position_lr_final = 0.0000016f,
        float position_lr_delay_mult = 0.01f,
        int position_lr_max_steps = 30'000,
        float feature_lr = 0.0025f,
        float opacity_lr = 0.05f,
        float scaling_lr = 0.005f,
        float rotation_lr = 0.001f,
        float percent_dense = 0.01f,
        float lambda_dssim = 0.2f,
        int densification_interval = 100,
        int opacity_reset_interval = 3000,
        int densify_from_iter = 500,
        int densify_until_iter = 15'000,
        float densify_grad_threshold = 0.0002f,
        float densify_abs_grad_threshold = 0.001f);

public:
    int iterations_;
    float position_lr_init_;
    float position_lr_final_;
    float position_lr_delay_mult_;
    int position_lr_max_steps_;
    float feature_lr_;
    float opacity_lr_;
    float scaling_lr_;
    float rotation_lr_;
    float percent_dense_;
    float lambda_dssim_;
    int densification_interval_;
    int opacity_reset_interval_;
    int densify_from_iter_;
    int densify_until_iter_;
    float densify_grad_threshold_;
    float densify_abs_grad_threshold_;
};
