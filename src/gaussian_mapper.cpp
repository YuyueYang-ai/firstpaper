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

#include <cctype>
#include <fstream>

#include <json/json.h>

#include "include/gaussian_mapper.h"
#include "include/phase2_residual_field.h"

namespace
{
std::string toLowerAscii(std::string value)
{
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string compactMetadataFormat(const std::filesystem::path& metadata_path)
{
    std::ifstream in(metadata_path);
    if (!in.is_open())
        throw std::runtime_error("Cannot open compact metadata at " + metadata_path.string());
    Json::CharReaderBuilder builder;
    Json::Value root;
    std::string errs;
    if (!Json::parseFromStream(builder, in, &root, &errs))
        throw std::runtime_error("Cannot parse compact metadata at " + metadata_path.string() + ": " + errs);
    return root.get("format", "").asString();
}
}

GaussianMapper::GaussianMapper(
    std::filesystem::path gaussian_config_file_path,
    std::filesystem::path result_dir,
    int seed,
    torch::DeviceType device_type)
    : initial_mapped_(false),
      interrupt_training_(false),
      stopped_(false),
      iteration_(0),
      ema_loss_for_log_(0.0f),
      loop_closure_iteration_(false),
      min_num_initial_map_kfs_(15UL),
      large_rot_th_(1e-1f),
      large_trans_th_(1e-2f),
      training_report_interval_(0)
{
    // Random seed
    std::srand(seed);
    torch::manual_seed(seed);

    // Device
    if (device_type == torch::kCUDA && torch::cuda::is_available()) {
        std::cout << "[Gaussian Mapper]CUDA available! Training on GPU." << std::endl;
        device_type_ = torch::kCUDA;
        model_params_.data_device_ = "cuda";
    }
    else {
        std::cout << "[Gaussian Mapper]Training on CPU." << std::endl;
        device_type_ = torch::kCPU;
        model_params_.data_device_ = "cpu";
    }

    result_dir_ = result_dir;
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    config_file_path_ = gaussian_config_file_path;
    readConfigFromFile(gaussian_config_file_path);

    std::vector<float> bg_color;
    if (model_params_.white_background_)
        bg_color = {1.0f, 1.0f, 1.0f};
    else
        bg_color = {0.0f, 0.0f, 0.0f};
    background_ = torch::tensor(bg_color,
                    torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    
    override_color_ = torch::empty(0, torch::TensorOptions().device(device_type_));

    // Initialize scene and model
    gaussians_ = std::make_shared<GaussianModel>(model_params_);
    scene_ = std::make_shared<GaussianScene>(model_params_);
}

void GaussianMapper::readConfigFromFile(std::filesystem::path cfg_path)
{
    cv::FileStorage settings_file(cfg_path.string().c_str(), cv::FileStorage::READ);
    if(!settings_file.isOpened()) {
       std::cerr << "[Gaussian Mapper]Failed to open settings file at: " << cfg_path << std::endl;
       exit(-1);
    }

    std::cout << "[Gaussian Mapper]Reading parameters from " << cfg_path << std::endl;
    std::unique_lock<std::mutex> lock(mutex_settings_);

    // Model parameters
    model_params_.sh_degree_ =
        settings_file["Model.sh_degree"].operator int();
    model_params_.resolution_ =
        settings_file["Model.resolution"].operator float();
    model_params_.white_background_ =
        (settings_file["Model.white_background"].operator int()) != 0;
    model_params_.eval_ =
        (settings_file["Model.eval"].operator int()) != 0;

    // Pipeline Parameters
    z_near_ =
        settings_file["Camera.z_near"].operator float();
    z_far_ =
        settings_file["Camera.z_far"].operator float();

    monocular_inactive_geo_densify_max_pixel_dist_ =
        settings_file["Monocular.inactive_geo_densify_max_pixel_dist"].operator float();
    stereo_min_disparity_ =
        settings_file["Stereo.min_disparity"].operator int();
    stereo_num_disparity_ =
        settings_file["Stereo.num_disparity"].operator int();
    RGBD_min_depth_ =
        settings_file["RGBD.min_depth"].operator float();
    RGBD_max_depth_ =
        settings_file["RGBD.max_depth"].operator float();

    inactive_geo_densify_ =
        (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;
    max_depth_cached_ =
        settings_file["Mapper.depth_cache"].operator int();
    min_num_initial_map_kfs_ = 
        static_cast<unsigned long>(settings_file["Mapper.min_num_initial_map_kfs"].operator int());
    new_keyframe_times_of_use_ = 
        settings_file["Mapper.new_keyframe_times_of_use"].operator int();
    local_BA_increased_times_of_use_ = 
        settings_file["Mapper.local_BA_increased_times_of_use"].operator int();
    loop_closure_increased_times_of_use_ = 
        settings_file["Mapper.loop_closure_increased_times_of_use_"].operator int();
    cull_keyframes_ =
        (settings_file["Mapper.cull_keyframes"].operator int()) != 0;
    large_rot_th_ =
        settings_file["Mapper.large_rotation_threshold"].operator float();
    large_trans_th_ =
        settings_file["Mapper.large_translation_threshold"].operator float();
    stable_num_iter_existence_ =
        settings_file["Mapper.stable_num_iter_existence"].operator int();

    pipe_params_.convert_SHs_ =
        (settings_file["Pipeline.convert_SHs"].operator int()) != 0;
    pipe_params_.compute_cov3D_ =
        (settings_file["Pipeline.compute_cov3D"].operator int()) != 0;

    do_gaus_pyramid_training_ =
        (settings_file["GausPyramid.do"].operator int()) != 0;
    num_gaus_pyramid_sub_levels_ =
        settings_file["GausPyramid.num_sub_levels"].operator int();
    int sub_level_times_of_use =
        settings_file["GausPyramid.sub_level_times_of_use"].operator int();
    kf_gaus_pyramid_times_of_use_.resize(num_gaus_pyramid_sub_levels_);
    kf_gaus_pyramid_factors_.resize(num_gaus_pyramid_sub_levels_);
    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        kf_gaus_pyramid_times_of_use_[l] = sub_level_times_of_use;
        kf_gaus_pyramid_factors_[l] = std::pow(0.5f, num_gaus_pyramid_sub_levels_ - l);
    }

    keyframe_record_interval_ = 
        settings_file["Record.keyframe_record_interval"].operator int();
    all_keyframes_record_interval_ = 
        settings_file["Record.all_keyframes_record_interval"].operator int();
    record_rendered_image_ = 
        (settings_file["Record.record_rendered_image"].operator int()) != 0;
    record_ground_truth_image_ = 
        (settings_file["Record.record_ground_truth_image"].operator int()) != 0;
    record_loss_image_ = 
        (settings_file["Record.record_loss_image"].operator int()) != 0;
    training_report_interval_ = 
        settings_file["Record.training_report_interval"].operator int();
    record_loop_ply_ =
        (settings_file["Record.record_loop_ply"].operator int()) != 0;

    // Optimization Parameters
    opt_params_.iterations_ =
        settings_file["Optimization.max_num_iterations"].operator int();
    opt_params_.position_lr_init_ =
        settings_file["Optimization.position_lr_init"].operator float();
    opt_params_.position_lr_final_ =
        settings_file["Optimization.position_lr_final"].operator float();
    opt_params_.position_lr_delay_mult_ =
        settings_file["Optimization.position_lr_delay_mult"].operator float();
    opt_params_.position_lr_max_steps_ =
        settings_file["Optimization.position_lr_max_steps"].operator int();
    opt_params_.feature_lr_ =
        settings_file["Optimization.feature_lr"].operator float();
    opt_params_.opacity_lr_ =
        settings_file["Optimization.opacity_lr"].operator float();
    opt_params_.scaling_lr_ =
        settings_file["Optimization.scaling_lr"].operator float();
    opt_params_.rotation_lr_ =
        settings_file["Optimization.rotation_lr"].operator float();

    opt_params_.percent_dense_ =
        settings_file["Optimization.percent_dense"].operator float();
    opt_params_.lambda_dssim_ =
        settings_file["Optimization.lambda_dssim"].operator float();
    opt_params_.densification_interval_ =
        settings_file["Optimization.densification_interval"].operator int();
    opt_params_.opacity_reset_interval_ =
        settings_file["Optimization.opacity_reset_interval"].operator int();
    opt_params_.densify_from_iter_ =
        settings_file["Optimization.densify_from_iter_"].operator int();
    opt_params_.densify_until_iter_ =
        settings_file["Optimization.densify_until_iter"].operator int();
    opt_params_.densify_grad_threshold_ =
        settings_file["Optimization.densify_grad_threshold"].operator float();

    prune_big_point_after_iter_ =
        settings_file["Optimization.prune_big_point_after_iter"].operator int();
    densify_min_opacity_ =
        settings_file["Optimization.densify_min_opacity"].operator float();
    prune_by_extent_ =
        (settings_file["Optimization.prune_by_extent"].operator int()) != 0;

    skip_bottom_ratio_ =
        settings_file["Optimization.skip_bottom_ratio"].operator float();

    const bool has_explicit_compression_mode = !settings_file["Compression.mode"].empty();
    const bool has_legacy_compression_options =
        !settings_file["Compression.save_compact_snapshot"].empty() ||
        !settings_file["Compression.late_stage_prune_interval"].empty() ||
        !settings_file["Compression.late_stage_prune_min_opacity"].empty() ||
        !settings_file["Compression.late_stage_prune_big_point_min_opacity"].empty() ||
        !settings_file["Compression.late_stage_prune_max_scaling_ratio"].empty() ||
        !settings_file["Compression.adaptive_sh_bandwidth_interval"].empty() ||
        !settings_file["Compression.adaptive_sh_energy_keep_ratio"].empty() ||
        !settings_file["Compression.adaptive_sh_min_opacity"].empty() ||
        !settings_file["Compression.adaptive_sh_export_only"].empty() ||
        !settings_file["Compression.export_sh_energy_keep_ratio"].empty() ||
        !settings_file["Compression.export_sh_min_opacity"].empty() ||
        !settings_file["Compression.export_sh_min_level"].empty() ||
        !settings_file["Compression.enable_export_prune"].empty() ||
        !settings_file["Compression.enable_sh_bandwidth"].empty() ||
        !settings_file["Compression.sort_by_morton"].empty() ||
        !settings_file["Compression.f_rest_blockwise_quant"].empty() ||
        !settings_file["Compression.f_rest_block_size"].empty() ||
        !settings_file["Compression.f_rest_locality_codec"].empty() ||
        !settings_file["Compression.f_rest_locality_high_sh_block_size"].empty() ||
        !settings_file["Compression.f_rest_locality_low_sh_block_size"].empty() ||
        !settings_file["Compression.f_rest_locality_int4_rel_mse_threshold"].empty() ||
        !settings_file["Compression.xyz_quant_bits"].empty() ||
        !settings_file["Compression.attribute_quant_bits"].empty() ||
        !settings_file["Compression.rotation_quant_bits"].empty();
    configureCompressionMode(
        parseCompressionMode(settings_file["Compression.mode"], has_legacy_compression_options));

    if (!has_explicit_compression_mode || compression_mode_ != CompressionMode::BASELINE) {
        if (!settings_file["Compression.save_compact_snapshot"].empty())
            save_compact_snapshot_ = (settings_file["Compression.save_compact_snapshot"].operator int()) != 0;
        if (!settings_file["Compression.late_stage_prune_interval"].empty())
            late_stage_prune_interval_ = settings_file["Compression.late_stage_prune_interval"].operator int();
        if (!settings_file["Compression.late_stage_prune_min_opacity"].empty())
            late_stage_prune_min_opacity_ = settings_file["Compression.late_stage_prune_min_opacity"].operator float();
        if (!settings_file["Compression.late_stage_prune_big_point_min_opacity"].empty())
            late_stage_prune_big_point_min_opacity_ = settings_file["Compression.late_stage_prune_big_point_min_opacity"].operator float();
        if (!settings_file["Compression.late_stage_prune_max_scaling_ratio"].empty())
            late_stage_prune_max_scaling_ratio_ = settings_file["Compression.late_stage_prune_max_scaling_ratio"].operator float();
        if (!settings_file["Compression.adaptive_sh_bandwidth_interval"].empty())
            adaptive_sh_bandwidth_interval_ = settings_file["Compression.adaptive_sh_bandwidth_interval"].operator int();
        if (!settings_file["Compression.adaptive_sh_energy_keep_ratio"].empty())
            adaptive_sh_energy_keep_ratio_ = settings_file["Compression.adaptive_sh_energy_keep_ratio"].operator float();
        if (!settings_file["Compression.adaptive_sh_min_opacity"].empty())
            adaptive_sh_min_opacity_ = settings_file["Compression.adaptive_sh_min_opacity"].operator float();
        if (!settings_file["Compression.adaptive_sh_export_only"].empty())
            adaptive_sh_export_only_ = (settings_file["Compression.adaptive_sh_export_only"].operator int()) != 0;
        if (!settings_file["Compression.export_sh_energy_keep_ratio"].empty())
            export_sh_energy_keep_ratio_ = settings_file["Compression.export_sh_energy_keep_ratio"].operator float();
        if (!settings_file["Compression.export_sh_min_opacity"].empty())
            export_sh_min_opacity_ = settings_file["Compression.export_sh_min_opacity"].operator float();
        if (!settings_file["Compression.export_sh_min_level"].empty())
            export_sh_min_level_ = settings_file["Compression.export_sh_min_level"].operator int();
        if (!settings_file["Compression.enable_export_prune"].empty())
            compact_export_options_.enable_export_prune = (settings_file["Compression.enable_export_prune"].operator int()) != 0;
        if (!settings_file["Compression.enable_sh_bandwidth"].empty())
            compact_export_options_.enable_sh_bandwidth = (settings_file["Compression.enable_sh_bandwidth"].operator int()) != 0;
        if (!settings_file["Compression.sort_by_morton"].empty())
            compact_export_options_.sort_by_morton = (settings_file["Compression.sort_by_morton"].operator int()) != 0;
        if (!settings_file["Compression.f_rest_blockwise_quant"].empty())
            f_rest_blockwise_quant_ = (settings_file["Compression.f_rest_blockwise_quant"].operator int()) != 0;
        if (!settings_file["Compression.f_rest_block_size"].empty())
            f_rest_block_size_ = settings_file["Compression.f_rest_block_size"].operator int();
        if (!settings_file["Compression.f_rest_locality_codec"].empty())
            f_rest_locality_codec_ = (settings_file["Compression.f_rest_locality_codec"].operator int()) != 0;
        if (!settings_file["Compression.f_rest_locality_high_sh_block_size"].empty())
            f_rest_locality_high_sh_block_size_ =
                settings_file["Compression.f_rest_locality_high_sh_block_size"].operator int();
        if (!settings_file["Compression.f_rest_locality_low_sh_block_size"].empty())
            f_rest_locality_low_sh_block_size_ =
                settings_file["Compression.f_rest_locality_low_sh_block_size"].operator int();
        if (!settings_file["Compression.f_rest_locality_int4_rel_mse_threshold"].empty())
            f_rest_locality_int4_rel_mse_threshold_ =
                settings_file["Compression.f_rest_locality_int4_rel_mse_threshold"].operator float();
        if (!settings_file["Compression.xyz_quant_bits"].empty())
            compact_export_options_.xyz_quant_bits = settings_file["Compression.xyz_quant_bits"].operator int();
        if (!settings_file["Compression.attribute_quant_bits"].empty())
            compact_export_options_.attribute_quant_bits = settings_file["Compression.attribute_quant_bits"].operator int();
        if (!settings_file["Compression.rotation_quant_bits"].empty())
            compact_export_options_.rotation_quant_bits = settings_file["Compression.rotation_quant_bits"].operator int();
    }

    compact_export_options_.prune_min_opacity = late_stage_prune_min_opacity_;
    compact_export_options_.prune_big_point_min_opacity = late_stage_prune_big_point_min_opacity_;
    compact_export_options_.prune_max_scaling_ratio = late_stage_prune_max_scaling_ratio_;
    compact_export_options_.sh_energy_keep_ratio = export_sh_energy_keep_ratio_;
    compact_export_options_.sh_min_opacity = export_sh_min_opacity_;
    compact_export_options_.sh_min_level = export_sh_min_level_;
    compact_export_options_.f_rest_blockwise_quant = f_rest_blockwise_quant_;
    compact_export_options_.f_rest_block_size = f_rest_block_size_;
    compact_export_options_.f_rest_locality_codec = f_rest_locality_codec_;
    compact_export_options_.f_rest_locality_high_sh_block_size = f_rest_locality_high_sh_block_size_;
    compact_export_options_.f_rest_locality_low_sh_block_size = f_rest_locality_low_sh_block_size_;
    compact_export_options_.f_rest_locality_int4_rel_mse_threshold = f_rest_locality_int4_rel_mse_threshold_;
    if (!settings_file["Phase2.enable"].empty())
        phase2_options_.enable = (settings_file["Phase2.enable"].operator int()) != 0;
    if (!settings_file["Phase2.save_frozen_snapshot"].empty())
        phase2_options_.save_frozen_snapshot = (settings_file["Phase2.save_frozen_snapshot"].operator int()) != 0;
    if (!settings_file["Phase2.freeze_topology_iter"].empty())
        phase2_options_.freeze_topology_iter = settings_file["Phase2.freeze_topology_iter"].operator int();
    if (!settings_file["Phase2.sort_by_morton"].empty())
        phase2_options_.sort_by_morton = (settings_file["Phase2.sort_by_morton"].operator int()) != 0;
    if (!settings_file["Phase2.normalize_xyz"].empty())
        phase2_options_.normalize_xyz = (settings_file["Phase2.normalize_xyz"].operator int()) != 0;
    if (!settings_file["Phase2.mask_features_rest_by_sh_level"].empty())
        phase2_options_.mask_features_rest_by_sh_level =
            (settings_file["Phase2.mask_features_rest_by_sh_level"].operator int()) != 0;
    phase2_options_.locality_high_sh_block_size = f_rest_locality_high_sh_block_size_;
    phase2_options_.locality_low_sh_block_size = f_rest_locality_low_sh_block_size_;
    if (!settings_file["Phase2.use_locality_base"].empty())
        phase2_options_.use_locality_base = (settings_file["Phase2.use_locality_base"].operator int()) != 0;
    if (!settings_file["Phase2.locality_high_sh_block_size"].empty())
        phase2_options_.locality_high_sh_block_size =
            settings_file["Phase2.locality_high_sh_block_size"].operator int();
    if (!settings_file["Phase2.locality_low_sh_block_size"].empty())
        phase2_options_.locality_low_sh_block_size =
            settings_file["Phase2.locality_low_sh_block_size"].operator int();
    if (!settings_file["Phase2HybridSelector.enable"].empty())
        phase2_options_.hybrid_selector.enable = (settings_file["Phase2HybridSelector.enable"].operator int()) != 0;
    if (!settings_file["Phase2HybridSelector.hard_point_ratio"].empty())
        phase2_options_.hybrid_selector.hard_point_ratio = settings_file["Phase2HybridSelector.hard_point_ratio"].operator float();
    if (!settings_file["Phase2HybridSelector.alpha"].empty())
        phase2_options_.hybrid_selector.alpha = settings_file["Phase2HybridSelector.alpha"].operator float();
    if (!settings_file["Phase2HybridSelector.beta"].empty())
        phase2_options_.hybrid_selector.beta = settings_file["Phase2HybridSelector.beta"].operator float();
    if (!settings_file["Phase2HybridSelector.gamma"].empty())
        phase2_options_.hybrid_selector.gamma = settings_file["Phase2HybridSelector.gamma"].operator float();
    if (!settings_file["Phase2HybridSelector.delta"].empty())
        phase2_options_.hybrid_selector.delta = settings_file["Phase2HybridSelector.delta"].operator float();
    phase2_options_.hybrid_selector.explicit_cost_int4_rel_mse_threshold =
        compact_export_options_.f_rest_locality_int4_rel_mse_threshold;
    if (!settings_file["Phase2HybridSelector.explicit_cost_int4_rel_mse_threshold"].empty())
        phase2_options_.hybrid_selector.explicit_cost_int4_rel_mse_threshold =
            settings_file["Phase2HybridSelector.explicit_cost_int4_rel_mse_threshold"].operator float();
    if (!settings_file["Phase2HybridSelector.min_hard_blocks"].empty())
        phase2_options_.hybrid_selector.min_hard_blocks = settings_file["Phase2HybridSelector.min_hard_blocks"].operator int();
    if (!settings_file["Phase2HybridSelector.max_hard_blocks"].empty())
        phase2_options_.hybrid_selector.max_hard_blocks = settings_file["Phase2HybridSelector.max_hard_blocks"].operator int();
    if (!settings_file["Phase2HybridSelector.save_debug_tensors"].empty())
        phase2_options_.hybrid_selector.save_debug_tensors =
            (settings_file["Phase2HybridSelector.save_debug_tensors"].operator int()) != 0;
    if (phase2_options_.freeze_topology_iter < 0)
        phase2_options_.freeze_topology_iter = opt_params_.densify_until_iter_;
    std::cout << "[Gaussian Mapper]Compression mode: " << compressionModeName(compression_mode_) << std::endl;
    if (compact_export_options_.enable_sh_bandwidth) {
        std::cout << "[Gaussian Mapper]Export SH bandwidth: keep_ratio=" << compact_export_options_.sh_energy_keep_ratio
                  << " min_opacity=" << compact_export_options_.sh_min_opacity
                  << " min_level=" << compact_export_options_.sh_min_level
                  << " export_only=" << (adaptive_sh_export_only_ ? "true" : "false")
                  << std::endl;
    }
    if (compact_export_options_.f_rest_blockwise_quant) {
        std::cout << "[Gaussian Mapper]f_rest blockwise quantization: block_size="
                  << compact_export_options_.f_rest_block_size << std::endl;
    }
    if (compact_export_options_.f_rest_locality_codec) {
        std::cout << "[Gaussian Mapper]f_rest locality codec: high_sh_block_size="
                  << compact_export_options_.f_rest_locality_high_sh_block_size
                  << " low_sh_block_size=" << compact_export_options_.f_rest_locality_low_sh_block_size
                  << " int4_rel_mse_threshold=" << compact_export_options_.f_rest_locality_int4_rel_mse_threshold
                  << std::endl;
    }
    if (phase2_options_.enable) {
        std::cout << "[Gaussian Mapper]Phase 2 residual field prep: freeze_iter="
                  << phase2_options_.freeze_topology_iter
                  << " save_snapshot=" << (phase2_options_.save_frozen_snapshot ? "true" : "false")
                  << " sort_by_morton=" << (phase2_options_.sort_by_morton ? "true" : "false")
                  << " normalize_xyz=" << (phase2_options_.normalize_xyz ? "true" : "false")
                  << " mask_f_rest=" << (phase2_options_.mask_features_rest_by_sh_level ? "true" : "false")
                  << " locality_base=" << (phase2_options_.use_locality_base ? "true" : "false")
                  << " high_block=" << phase2_options_.locality_high_sh_block_size
                  << " low_block=" << phase2_options_.locality_low_sh_block_size
                  << " hybrid_selector=" << (phase2_options_.hybrid_selector.enable ? "true" : "false")
                  << " hybrid_hard_ratio=" << phase2_options_.hybrid_selector.hard_point_ratio
                  << " selector_weights=("
                  << phase2_options_.hybrid_selector.alpha << ","
                  << phase2_options_.hybrid_selector.beta << ","
                  << phase2_options_.hybrid_selector.gamma << ","
                  << phase2_options_.hybrid_selector.delta << ")"
                  << " selector_int4_thresh=" << phase2_options_.hybrid_selector.explicit_cost_int4_rel_mse_threshold
                  << std::endl;
    }

    // Viewer Parameters
    rendered_image_viewer_scale_ =
        settings_file["GaussianViewer.image_scale"].operator float();
    rendered_image_viewer_scale_main_ =
        settings_file["GaussianViewer.image_scale_main"].operator float();
}

void GaussianMapper::trainSfmPcd()
{
    // Prepare multi resolution images for training
    for (auto& kfit : scene_->keyframes()) {
        auto pkf = kfit.second;
        increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());
        if (device_type_ == torch::kCUDA) {
            cv::cuda::GpuMat img_gpu;
            img_gpu.upload(pkf->img_undist_);
            pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                cv::cuda::GpuMat img_resized;
                cv::cuda::resize(img_gpu, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                pkf->gaus_pyramid_original_image_[l] =
                    tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
            }
        }
        else {
            pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                cv::Mat img_resized;
                cv::resize(pkf->img_undist_, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                pkf->gaus_pyramid_original_image_[l] =
                    tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
            }
        }
    }

    // Prepare for training
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
        gaussians_->createFromPcd(scene_->cached_point_cloud_, scene_->cameras_extent_);
        std::unique_lock<std::mutex> lock(mutex_settings_);
        gaussians_->trainingSetup(opt_params_);
        this->initial_mapped_ = true;
    }

    // Main loop: gaussian splatting training
    while (!isStopped()) {
        // Invoke training once
        trainForOneIteration();

        if (getIteration() >= opt_params_.iterations_)
            break;
    }

    // Tail gaussian optimization
    int densify_interval = densifyInterval();
    int n_delay_iters = densify_interval * 0.8;
    while (getIteration() % densify_interval <= n_delay_iters || isKeepingTraining()) {
        trainForOneIteration();
        densify_interval = densifyInterval();
        n_delay_iters = densify_interval * 0.8;
    }

    // Save and clear
    renderAndRecordAllKeyframes("_shutdown");
    savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

    signalStop();
}

/**
 * @brief The training iteration body
 * 
 */
void GaussianMapper::trainForOneIteration()
{
    increaseIteration(1);
    auto iter_start_timing = std::chrono::steady_clock::now();

    // Pick a random Camera
    std::shared_ptr<GaussianKeyframe> viewpoint_cam = useOneRandomSlidingWindowKeyframe();
    if (!viewpoint_cam) {
        increaseIteration(-1);
        return;
    }

    writeKeyframeUsedTimes(result_dir_ / "used_times");

    int training_level = num_gaus_pyramid_sub_levels_;
    int image_height, image_width;
    torch::Tensor gt_image, mask;
    if (isdoingGausPyramidTraining())
        training_level = viewpoint_cam->getCurrentGausPyramidLevel();
    if (training_level == num_gaus_pyramid_sub_levels_) {
        image_height = viewpoint_cam->image_height_;
        image_width = viewpoint_cam->image_width_;
        gt_image = viewpoint_cam->original_image_.cuda();
        mask = undistort_mask_[viewpoint_cam->camera_id_];
    }
    else {
        image_height = viewpoint_cam->gaus_pyramid_height_[training_level];
        image_width = viewpoint_cam->gaus_pyramid_width_[training_level];
        gt_image = viewpoint_cam->gaus_pyramid_original_image_[training_level].cuda();
        mask = scene_->cameras_.at(viewpoint_cam->camera_id_).gaus_pyramid_undistort_mask_[training_level];
    }

    // Mutex lock for usage of the gaussian model
    std::unique_lock<std::mutex> lock_render(mutex_render_);

    // Every 1000 its we increase the levels of SH up to a maximum degree
    if (getIteration() % 1000 == 0 && default_sh_ < model_params_.sh_degree_)
        default_sh_ += 1;
    // if (isdoingGausPyramidTraining())
    //     gaussians_->setShDegree(training_level);
    // else
        gaussians_->setShDegree(default_sh_);

    // Update learning rate
    gaussians_->updateLearningRate(getIteration());
    gaussians_->setFeatureLearningRate(featureLearningRate());
    gaussians_->setOpacityLearningRate(opacityLearningRate());
    gaussians_->setScalingLearningRate(scalingLearningRate());
    gaussians_->setRotationLearningRate(rotationLearningRate());

    // Render
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> render_pkg;
    switch ((Camera::CameraModelType)viewpoint_cam->camera_model_id_)
    {
    case Camera::CameraModelType::PINHOLE:
    {
        render_pkg = GaussianRenderer::render(
            viewpoint_cam,
            image_height,
            image_width,
            gaussians_,
            pipe_params_,
            background_,
            override_color_,
            false
        );
    }
    break;
    case Camera::CameraModelType::LONLAT:
    {
        render_pkg = GaussianRenderer::renderLonlat(
            viewpoint_cam,
            image_height,
            image_width,
            gaussians_,
            pipe_params_,
            background_,
            override_color_
        );
    }
    break;
    default:
    {
        // TODO: use std::source_location to get the name of current function
        throw std::runtime_error("[GaussianMapper::trainForOneIteration]Invalid camera model type.");
    }
        break;
    }

    auto rendered_image = std::get<0>(render_pkg);
    auto viewspace_point_tensor = std::get<1>(render_pkg);
    auto visibility_filter = std::get<2>(render_pkg);
    auto radii = std::get<3>(render_pkg);

    // Get rid of black edges caused by undistortion
    torch::Tensor masked_image = rendered_image * mask;

    // Loss
    torch::Tensor Ll1, loss;
    if (skip_bottom_ratio_ > 0.0f) {
        // Skip bottom if needed (esspecially for panoramas in real world scenes, e.g. 360Roam)
        int pix_skip_bottom = (int)std::round(image_height * skip_bottom_ratio_);
        torch::Tensor masked_image_skip_bottom = masked_image.index(
            {torch::indexing::Slice(), torch::indexing::Slice(0, -pix_skip_bottom), torch::indexing::Slice()});
        torch::Tensor gt_image_skip_bottom = gt_image.index(
            {torch::indexing::Slice(), torch::indexing::Slice(0, -pix_skip_bottom), torch::indexing::Slice()});

        Ll1 = loss_utils::l1_loss(masked_image_skip_bottom, gt_image_skip_bottom);
        float lambda_dssim = lambdaDssim();
        loss = (1.0 - lambda_dssim) * Ll1
                + lambda_dssim * (1.0 - loss_utils::ssim(masked_image_skip_bottom, gt_image_skip_bottom, device_type_));
    }
    else {
        Ll1 = loss_utils::l1_loss(masked_image, gt_image);
        float lambda_dssim = lambdaDssim();
        loss = (1.0 - lambda_dssim) * Ll1
                + lambda_dssim * (1.0 - loss_utils::ssim(masked_image, gt_image, device_type_));
    }
    loss.backward();

    torch::cuda::synchronize();

    {
        torch::NoGradGuard no_grad;
        ema_loss_for_log_ = 0.4f * loss.item().toFloat() + 0.6 * ema_loss_for_log_;

        if (phase2_options_.enable
            && !phase2_topology_frozen_
            && getIteration() >= phase2_options_.freeze_topology_iter) {
            phase2_topology_frozen_ = true;
            late_stage_prune_interval_ = 0;
            adaptive_sh_bandwidth_interval_ = 0;
            std::cout << "[Gaussian Mapper]Phase 2 topology frozen at iteration "
                      << getIteration() << std::endl;
            if (phase2_options_.save_frozen_snapshot)
                savePhase2FrozenPackage(result_dir_ / std::to_string(getIteration()));
        }

        if (keyframe_record_interval_ &&
            getIteration() % keyframe_record_interval_ == 0)
            recordKeyframeRendered(masked_image, gt_image, viewpoint_cam->fid_, result_dir_, result_dir_, result_dir_);

        // Densification
        if (getIteration() < opt_params_.densify_until_iter_) {
            // Keep track of max radii in image-space for pruning
            gaussians_->max_radii2D_.index_put_(
                {visibility_filter},
                torch::max(gaussians_->max_radii2D_.index({visibility_filter}),
                            radii.index({visibility_filter})));
            // if (!isdoingGausPyramidTraining() || training_level < num_gaus_pyramid_sub_levels_)
                gaussians_->addDensificationStats(viewspace_point_tensor, visibility_filter);

            if ((getIteration() > opt_params_.densify_from_iter_) &&
                (getIteration() % densifyInterval()== 0)) {
                int size_threshold = (getIteration() > prune_big_point_after_iter_) ? 20 : 0;
                gaussians_->densifyAndPrune(
                    densifyGradThreshold(),
                    densify_min_opacity_,//0.005,//
                    scene_->cameras_extent_,
                    size_threshold,
                    prune_by_extent_
                );
            }

                if (opacityResetInterval()
                    && (getIteration() % opacityResetInterval() == 0
                        ||(model_params_.white_background_ && getIteration() == opt_params_.densify_from_iter_)))
                    gaussians_->resetOpacity();
        }
        else {
            if (late_stage_prune_interval_ > 0 && getIteration() % late_stage_prune_interval_ == 0) {
                gaussians_->pruneLowImportanceGaussians(
                    late_stage_prune_min_opacity_,
                    late_stage_prune_big_point_min_opacity_,
                    late_stage_prune_max_scaling_ratio_,
                    scene_->cameras_extent_);
            }
            if (!adaptive_sh_export_only_
                && adaptive_sh_bandwidth_interval_ > 0
                && getIteration() % adaptive_sh_bandwidth_interval_ == 0) {
                gaussians_->updateAdaptiveShBandwidth(
                    adaptive_sh_energy_keep_ratio_,
                    adaptive_sh_min_opacity_);
            }
        }

        auto iter_end_timing = std::chrono::steady_clock::now();
        auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        iter_end_timing - iter_start_timing).count();

        // Log and save
        if (training_report_interval_ && (getIteration() % training_report_interval_ == 0))
            GaussianTrainer::trainingReport(
                getIteration(),
                opt_params_.iterations_,
                Ll1,
                loss,
                ema_loss_for_log_,
                loss_utils::l1_loss,
                iter_time,
                *gaussians_,
                *scene_,
                pipe_params_,
                background_
            );
        if ((all_keyframes_record_interval_ && getIteration() % all_keyframes_record_interval_ == 0)
            // || loop_closure_iteration_
            )
        {
            renderAndRecordAllKeyframes();
            savePly(result_dir_ / std::to_string(getIteration()) / "ply");
        }

        if (loop_closure_iteration_)
            loop_closure_iteration_ = false;

        // Optimizer step
        if (getIteration() < opt_params_.iterations_) {
            gaussians_->optimizer_->step();
            gaussians_->optimizer_->zero_grad(true);
        }
    }
}

bool GaussianMapper::isStopped()
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    return this->stopped_;
}

void GaussianMapper::configureCompressionMode(CompressionMode mode)
{
    compression_mode_ = mode;

    late_stage_prune_min_opacity_ = 0.008f;
    late_stage_prune_big_point_min_opacity_ = 0.02f;
    late_stage_prune_max_scaling_ratio_ = 0.15f;
    adaptive_sh_energy_keep_ratio_ = 0.995f;
    adaptive_sh_min_opacity_ = 0.01f;
    adaptive_sh_export_only_ = false;
    export_sh_energy_keep_ratio_ = adaptive_sh_energy_keep_ratio_;
    export_sh_min_opacity_ = adaptive_sh_min_opacity_;
    export_sh_min_level_ = 0;
    f_rest_blockwise_quant_ = false;
    f_rest_block_size_ = 128;
    f_rest_locality_codec_ = false;
    f_rest_locality_high_sh_block_size_ = 64;
    f_rest_locality_low_sh_block_size_ = 128;
    f_rest_locality_int4_rel_mse_threshold_ = 0.02f;
    phase2_options_ = Phase2ResidualFieldOptions();
    phase2_topology_frozen_ = false;

    compact_export_options_ = CompactExportOptions();
    compact_export_options_.prune_min_opacity = late_stage_prune_min_opacity_;
    compact_export_options_.prune_big_point_min_opacity = late_stage_prune_big_point_min_opacity_;
    compact_export_options_.prune_max_scaling_ratio = late_stage_prune_max_scaling_ratio_;
    compact_export_options_.sh_energy_keep_ratio = export_sh_energy_keep_ratio_;
    compact_export_options_.sh_min_opacity = export_sh_min_opacity_;
    compact_export_options_.sh_min_level = export_sh_min_level_;
    compact_export_options_.f_rest_blockwise_quant = f_rest_blockwise_quant_;
    compact_export_options_.f_rest_block_size = f_rest_block_size_;
    compact_export_options_.f_rest_locality_codec = f_rest_locality_codec_;
    compact_export_options_.f_rest_locality_high_sh_block_size = f_rest_locality_high_sh_block_size_;
    compact_export_options_.f_rest_locality_low_sh_block_size = f_rest_locality_low_sh_block_size_;
    compact_export_options_.f_rest_locality_int4_rel_mse_threshold = f_rest_locality_int4_rel_mse_threshold_;

    if (mode == CompressionMode::COMPACT || mode == CompressionMode::COMPACT_PHASE2) {
        save_compact_snapshot_ = true;
        late_stage_prune_interval_ = 1000;
        adaptive_sh_bandwidth_interval_ = 500;
        compact_export_options_.enable_export_prune = true;
        compact_export_options_.enable_sh_bandwidth = true;
        compact_export_options_.sort_by_morton = true;
        if (mode == CompressionMode::COMPACT_PHASE2) {
            f_rest_locality_codec_ = true;
            compact_export_options_.f_rest_locality_codec = true;
            compact_export_options_.f_rest_locality_high_sh_block_size = f_rest_locality_high_sh_block_size_;
            compact_export_options_.f_rest_locality_low_sh_block_size = f_rest_locality_low_sh_block_size_;
            compact_export_options_.f_rest_locality_int4_rel_mse_threshold =
                f_rest_locality_int4_rel_mse_threshold_;
            phase2_options_.enable = true;
            phase2_options_.save_frozen_snapshot = true;
            phase2_options_.sort_by_morton = true;
            phase2_options_.normalize_xyz = true;
            phase2_options_.mask_features_rest_by_sh_level = true;
            phase2_options_.use_locality_base = true;
            phase2_options_.locality_high_sh_block_size = f_rest_locality_high_sh_block_size_;
            phase2_options_.locality_low_sh_block_size = f_rest_locality_low_sh_block_size_;
        }
        return;
    }

    save_compact_snapshot_ = false;
    late_stage_prune_interval_ = 0;
    adaptive_sh_bandwidth_interval_ = 0;
    compact_export_options_.enable_export_prune = false;
    compact_export_options_.enable_sh_bandwidth = false;
    compact_export_options_.sort_by_morton = false;
    compact_export_options_.f_rest_blockwise_quant = false;
    compact_export_options_.f_rest_locality_codec = false;
}

CompressionMode GaussianMapper::parseCompressionMode(
    const cv::FileNode& mode_node,
    bool has_legacy_options)
{
    if (mode_node.empty())
        return has_legacy_options ? CompressionMode::COMPACT : CompressionMode::BASELINE;

    if (mode_node.isInt()) {
        const int mode_value = mode_node.operator int();
        if (mode_value == 0)
            return CompressionMode::BASELINE;
        if (mode_value == 1)
            return CompressionMode::COMPACT;
        if (mode_value == 2)
            return CompressionMode::COMPACT_PHASE2;
    }

    const std::string mode_value = toLowerAscii(mode_node.string());
    if (mode_value == "baseline" || mode_value == "base" || mode_value == "0")
        return CompressionMode::BASELINE;
    if (mode_value == "compact" || mode_value == "compressed" || mode_value == "1")
        return CompressionMode::COMPACT;
    if (mode_value == "compact_phase2" || mode_value == "compact-phase2" || mode_value == "phase2" || mode_value == "2")
        return CompressionMode::COMPACT_PHASE2;

    throw std::runtime_error("Unsupported Compression.mode: " + mode_node.string());
}

bool GaussianMapper::looksLikeCompactResultPath(const std::filesystem::path& result_path)
{
    if (result_path.empty())
        return false;
    if (std::filesystem::is_directory(result_path))
        return std::filesystem::exists(result_path / "metadata.json");
    return result_path.filename() == "metadata.json";
}

void GaussianMapper::signalStop(const bool going_to_stop)
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    this->stopped_ = going_to_stop;
}

void GaussianMapper::generateKfidRandomShuffle()
{
// if (viewpoint_sliding_window_.empty())
//     return;

// std::size_t sliding_window_size = viewpoint_sliding_window_.size();
// kfid_shuffle_.resize(sliding_window_size);
// std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
// std::mt19937 g(rd_());
// std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

    if (scene_->keyframes().empty())
        return;

    std::size_t nkfs = scene_->keyframes().size();
    kfid_shuffle_.resize(nkfs);
    std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
    std::mt19937 g(rd_());
    std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

    kfid_shuffled_ = true;
}

std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomSlidingWindowKeyframe()
{
// auto t1 = std::chrono::steady_clock::now();
    if (scene_->keyframes().empty())
        return nullptr;

    if (!kfid_shuffled_)
        generateKfidRandomShuffle();

    std::shared_ptr<GaussianKeyframe> viewpoint_cam = nullptr;
    int random_cam_idx;

    if (kfid_shuffled_) {
        int start_shuffle_idx = kfid_shuffle_idx_;
        do {
            // Next shuffled idx
            ++kfid_shuffle_idx_;
            if (kfid_shuffle_idx_ >= kfid_shuffle_.size())
                kfid_shuffle_idx_ = 0;
            // Add 1 time of use to all kfs if they are all unavalible, then shuffle again
            if (kfid_shuffle_idx_ == start_shuffle_idx) {
                for (auto& kfit : scene_->keyframes())
                    increaseKeyframeTimesOfUse(kfit.second, 1);
                generateKfidRandomShuffle();
            }

            // Get viewpoint kf
            random_cam_idx = kfid_shuffle_[kfid_shuffle_idx_];
            auto random_cam_it = scene_->keyframes().begin();
            for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
                ++random_cam_it;
            viewpoint_cam = (*random_cam_it).second;
        } while (viewpoint_cam->remaining_times_of_use_ <= 0);
    }

    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];
    
    // Handle times of use
    --(viewpoint_cam->remaining_times_of_use_);

// auto t2 = std::chrono::steady_clock::now();
// auto t21 = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
// std::cout<<t21 <<" ns"<<std::endl;
    return viewpoint_cam;
}

std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomKeyframe()
{
    if (scene_->keyframes().empty())
        return nullptr;

    // Get randomly
    int nkfs = static_cast<int>(scene_->keyframes().size());
    int random_cam_idx = std::rand() / ((RAND_MAX + 1u) / nkfs);
    auto random_cam_it = scene_->keyframes().begin();
    for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
        ++random_cam_it;
    std::shared_ptr<GaussianKeyframe> viewpoint_cam = (*random_cam_it).second;

    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];

    return viewpoint_cam;
}

void GaussianMapper::increaseKeyframeTimesOfUse(
    std::shared_ptr<GaussianKeyframe> pkf,
    int times)
{
    pkf->remaining_times_of_use_ += times;
}

// bool GaussianMapper::needInterruptTraining()
// {
//     std::unique_lock<std::mutex> lock_status(this->mutex_status_);
//     return this->interrupt_training_;
// }

// void GaussianMapper::setInterruptTraining(const bool interrupt_training)
// {
//     std::unique_lock<std::mutex> lock_status(this->mutex_status_);
//     this->interrupt_training_ = interrupt_training;
// }

void GaussianMapper::recordKeyframeRendered(
        torch::Tensor &rendered,
        torch::Tensor &ground_truth,
        unsigned long kfid,
        std::filesystem::path result_img_dir,
        std::filesystem::path result_gt_dir,
        std::filesystem::path result_loss_dir,
        std::string name_suffix)
{
    if (record_rendered_image_) {
        auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered);
        cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
        image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_img_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".jpg"), image_cv);
    }

    if (record_ground_truth_image_) {
        auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(ground_truth);
        cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
        gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_gt_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_gt.jpg"), gt_image_cv);
    }

    if (record_loss_image_) {
        torch::Tensor loss_tensor = torch::abs(rendered - ground_truth);
        auto loss_image_cv = tensor_utils::torchTensor2CvMat_Float32(loss_tensor);
        cv::cvtColor(loss_image_cv, loss_image_cv, CV_RGB2BGR);
        loss_image_cv.convertTo(loss_image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_loss_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_loss.jpg"), loss_image_cv);
    }
}

cv::Mat GaussianMapper::renderFromPose(
    const Sophus::SE3f &Tcw,
    const int width,
    const int height,
    const bool main_vision,
    Camera::CameraModelType camera_type,
    bool render_depth)
{
    if (!initial_mapped_)
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>();
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    pkf->setPose(
        Tcw.unit_quaternion().cast<double>(),
        Tcw.translation().cast<double>());
    try {
        // Camera
        Camera& camera = scene_->cameras_.at(viewer_camera_id_);
        pkf->setCameraParams(camera);
        // Transformations
        pkf->computeTransformTensors();
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::renderFromPose]KeyFrame Camera not found!");
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> render_pkg;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        // Render
        switch (camera_type)
        {
        case Camera::CameraModelType::PINHOLE:
        {
            render_pkg = GaussianRenderer::render(
                pkf,
                height,
                width,
                gaussians_,
                pipe_params_,
                background_,
                override_color_,
                render_depth
            );
        }
        break;
        case Camera::CameraModelType::LONLAT:
        {
            render_pkg = GaussianRenderer::renderLonlat(
                pkf,
                height,
                width,
                gaussians_,
                pipe_params_,
                background_,
                override_color_
            );
        }
        break;
        default:
        {
            throw std::runtime_error("[GaussianMapper::renderFromPose]Invalid camera model type.");
        }
        break;
        }
    }

    // Result
    torch::Tensor masked_image;
    if (main_vision)
        masked_image = std::get<0>(render_pkg) * viewer_main_undistort_mask_[pkf->camera_id_];
    else
        masked_image = std::get<0>(render_pkg) * viewer_sub_undistort_mask_[pkf->camera_id_];
    return tensor_utils::torchTensor2CvMat_Float32(masked_image);
}

void GaussianMapper::renderAndRecordKeyframe(
    std::shared_ptr<GaussianKeyframe> pkf,
    float &dssim,
    float &psnr,
    float &psnr_gs,
    double &render_time,
    std::filesystem::path result_img_dir,
    std::filesystem::path result_gt_dir,
    std::filesystem::path result_loss_dir,
    std::string name_suffix,
    bool render_depth)
{
    auto start_timing = std::chrono::steady_clock::now();
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> render_pkg;
    // Render
    switch (pkf->camera_model_id_)
    {
    case Camera::CameraModelType::PINHOLE:
    {
        render_pkg = GaussianRenderer::render(
            pkf,
            pkf->image_height_,
            pkf->image_width_,
            gaussians_,
            pipe_params_,
            background_,
            override_color_,
            render_depth
        );
    }
    break;
    case Camera::CameraModelType::LONLAT:
    {
        render_pkg = GaussianRenderer::renderLonlat(
            pkf,
            pkf->image_height_,
            pkf->image_width_,
            gaussians_,
            pipe_params_,
            background_,
            override_color_
        );
    }
    break;
    default:
    {
        throw std::runtime_error("[GaussianMapper::renderAndRecordKeyframe]Invalid camera model type.");
    }
    break;
    }

    auto rendered_image = std::get<0>(render_pkg);
    torch::Tensor masked_image = rendered_image * undistort_mask_[pkf->camera_id_];
    torch::cuda::synchronize();
    auto end_timing = std::chrono::steady_clock::now();
    auto render_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_timing - start_timing).count();
    render_time = 1e-6 * render_time_ns;
    auto gt_image = pkf->original_image_;

    if (skip_bottom_ratio_ > 0.0f) {
        int pix_skip_bottom = (int)std::round(pkf->image_height_ * skip_bottom_ratio_);
        masked_image = masked_image.index(
            {torch::indexing::Slice(), torch::indexing::Slice(0, -pix_skip_bottom), torch::indexing::Slice()});
        gt_image = gt_image.index(
            {torch::indexing::Slice(), torch::indexing::Slice(0, -pix_skip_bottom), torch::indexing::Slice()});
    }

    dssim = loss_utils::ssim(masked_image, gt_image, device_type_).item().toFloat();
    psnr = loss_utils::psnr(masked_image, gt_image).item().toFloat();
    psnr_gs = loss_utils::psnr_gaussian_splatting(masked_image, gt_image).item().toFloat();

    recordKeyframeRendered(masked_image, gt_image, pkf->fid_, result_img_dir, result_gt_dir, result_loss_dir, name_suffix);    
}

void GaussianMapper::renderAndRecordAllKeyframes(
    std::string name_suffix)
{
    std::filesystem::path result_dir = result_dir_ / (std::to_string(getIteration()) + name_suffix);
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path image_dir = result_dir / "image";
    if (record_rendered_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_dir);

    std::filesystem::path image_gt_dir = result_dir / "image_gt";
    if (record_ground_truth_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_gt_dir);

    std::filesystem::path image_loss_dir = result_dir / "image_loss";
    if (record_loss_image_) {
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_loss_dir);
    }

    std::filesystem::path render_time_path = result_dir / "render_time.txt";
    std::ofstream out_time(render_time_path);
    out_time << "##[Gaussian Mapper]Render time statistics: keyframe id, time(milliseconds)" << std::endl;

    std::filesystem::path dssim_path = result_dir / "dssim.txt";
    std::ofstream out_dssim(dssim_path);
    out_dssim << "##[Gaussian Mapper]keyframe id, dssim" << std::endl;

    std::filesystem::path psnr_path = result_dir / "psnr.txt";
    std::ofstream out_psnr(psnr_path);
    out_psnr << "##[Gaussian Mapper]keyframe id, psnr" << std::endl;

    std::filesystem::path psnr_gs_path = result_dir / "psnr_gaussian_splatting.txt";
    std::ofstream out_psnr_gs(psnr_gs_path);
    out_psnr_gs << "##[Gaussian Mapper]keyframe id, psnr_gaussian_splatting" << std::endl;

    std::size_t nkfs = scene_->keyframes().size();
    auto kfit = scene_->keyframes().begin();
    float dssim, psnr, psnr_gs;
    double render_time;
    for (std::size_t i = 0; i < nkfs; ++i) {
        renderAndRecordKeyframe((*kfit).second, dssim, psnr, psnr_gs, render_time, image_dir, image_gt_dir, image_loss_dir);
        out_time << (*kfit).first << " " << std::fixed << std::setprecision(8) << render_time << std::endl;

        out_dssim   << (*kfit).first << " " << std::fixed << std::setprecision(10) << dssim   << std::endl;
        out_psnr    << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr    << std::endl;
        out_psnr_gs << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr_gs << std::endl;

        ++kfit;
    }
}

void GaussianMapper::savePly(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    keyframesToJson(result_dir);
    saveModelParams(result_dir);

    std::filesystem::path ply_dir = result_dir / "point_cloud";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    ply_dir = ply_dir / ("iteration_" + std::to_string(getIteration()));
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    gaussians_->savePly(ply_dir / "point_cloud.ply");

    if (save_compact_snapshot_)
        saveCompact(result_dir);
    if (phase2_options_.enable && phase2_topology_frozen_ && phase2_options_.save_frozen_snapshot)
        savePhase2FrozenPackage(result_dir);
}

void GaussianMapper::saveCompact(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path compact_dir = result_dir / "compact";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(compact_dir)

    compact_dir = compact_dir / ("iteration_" + std::to_string(getIteration()));
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(compact_dir)

    CompactExportOptions export_options = compact_export_options_;
    if (scene_->cameras_extent_ <= 0.0f)
        export_options.enable_export_prune = false;

    gaussians_->saveCompact(compact_dir, scene_->cameras_extent_, export_options);
}

void GaussianMapper::savePhase2FrozenPackage(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path phase2_dir = result_dir / "phase2";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(phase2_dir)

    phase2_dir = phase2_dir / ("iteration_" + std::to_string(getIteration()));
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(phase2_dir)

    DecodedGaussianTensors decoded;
    decoded.max_sh_degree = gaussians_->max_sh_degree_;
    decoded.active_sh_degree = gaussians_->active_sh_degree_;
    decoded.xyz = gaussians_->xyz_.detach();
    decoded.features_dc = gaussians_->features_dc_.detach();
    decoded.features_rest = gaussians_->features_rest_.detach();
    decoded.opacity = gaussians_->opacity_.detach();
    decoded.scaling = gaussians_->scaling_.detach();
    decoded.rotation = gaussians_->rotation_.detach();
    decoded.sh_levels = (gaussians_->sh_levels_.defined() && gaussians_->sh_levels_.numel() != 0)
                            ? gaussians_->sh_levels_.detach()
                            : torch::full(
                                  {gaussians_->xyz_.size(0)},
                                  gaussians_->max_sh_degree_,
                                  torch::TensorOptions().dtype(torch::kInt32).device(gaussians_->xyz_.device()));

    phase2_residual_field::saveFrozenPackage(
        decoded,
        phase2_dir,
        scene_->cameras_extent_,
        phase2_options_);
}

void GaussianMapper::keyframesToJson(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path result_path = result_dir / "cameras.json";
    std::ofstream out_stream;
    out_stream.open(result_path);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json file at " + result_path.string());

    Json::Value json_root;
    Json::StreamWriterBuilder builder;
    const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

    int i = 0;
    for (const auto& kfit : scene_->keyframes()) {
        const auto pkf = kfit.second;
        Eigen::Matrix4f Rt;
        Rt.setZero();
        Eigen::Matrix3f R = pkf->R_quaternion_.toRotationMatrix().cast<float>();
        Rt.topLeftCorner<3, 3>() = R;
        Eigen::Vector3f t = pkf->t_.cast<float>();
        Rt.topRightCorner<3, 1>() = t;
        Rt(3, 3) = 1.0f;

        Eigen::Matrix4f Twc = Rt.inverse();
        Eigen::Vector3f pos = Twc.block<3, 1>(0, 3);
        Eigen::Matrix3f rot = Twc.block<3, 3>(0, 0);

        Json::Value json_kf;
        json_kf["id"] = static_cast<Json::Value::UInt64>(pkf->fid_);
        json_kf["img_name"] = pkf->img_filename_; //(std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_));
        json_kf["width"] = pkf->image_width_;
        json_kf["height"] = pkf->image_height_;

        json_kf["position"][0] = pos.x();
        json_kf["position"][1] = pos.y();
        json_kf["position"][2] = pos.z();

        json_kf["rotation"][0][0] = rot(0, 0);
        json_kf["rotation"][0][1] = rot(0, 1);
        json_kf["rotation"][0][2] = rot(0, 2);
        json_kf["rotation"][1][0] = rot(1, 0);
        json_kf["rotation"][1][1] = rot(1, 1);
        json_kf["rotation"][1][2] = rot(1, 2);
        json_kf["rotation"][2][0] = rot(2, 0);
        json_kf["rotation"][2][1] = rot(2, 1);
        json_kf["rotation"][2][2] = rot(2, 2);

        json_kf["fy"] = graphics_utils::fov2focal(pkf->FoVy_, pkf->image_height_);
        json_kf["fx"] = graphics_utils::fov2focal(pkf->FoVx_, pkf->image_width_);

        json_root[i] = Json::Value(json_kf);
        ++i;
    }

    writer->write(json_root, &out_stream);
}

void GaussianMapper::saveModelParams(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path result_path = result_dir / "cfg_args";
    std::ofstream out_stream;
    out_stream.open(result_path);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open file at " + result_path.string());

    out_stream << "Namespace("
               << "eval=" << (model_params_.eval_ ? "True" : "False") << ", "
               << "images=" << "\'" << model_params_.images_ << "\', "
               << "model_path=" << "\'" << model_params_.model_path_.string() << "\', "
               << "resolution=" << model_params_.resolution_ << ", "
               << "sh_degree=" << model_params_.sh_degree_ << ", "
               << "source_path=" << "\'" << model_params_.source_path_.string() << "\', "
               << "white_background=" << (model_params_.white_background_ ? "True" : "False") << ", "
               << ")";

    out_stream.close();
}

void GaussianMapper::writeKeyframeUsedTimes(std::filesystem::path result_dir, std::string name_suffix)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path result_path = result_dir / ("keyframe_used_times" + name_suffix + ".txt");
    std::ofstream out_stream;
    out_stream.open(result_path, std::ios::app);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json at " + result_path.string());

    out_stream << "##[Gaussian Mapper]Iteration " << getIteration() << " keyframe id, used times, remaining times:\n";
    for (const auto& used_times_it : kfs_used_times_)
        out_stream << used_times_it.first << " "
                   << used_times_it.second << " "
                   << scene_->keyframes().at(used_times_it.first)->remaining_times_of_use_
                   << "\n";
    out_stream << "##=========================================" <<std::endl;

    out_stream.close();
}

int GaussianMapper::getIteration()
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    return iteration_;
}
void GaussianMapper::increaseIteration(const int inc)
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    iteration_ += inc;
}

float GaussianMapper::positionLearningRateInit()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.position_lr_init_;
}
float GaussianMapper::featureLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.feature_lr_;
}
float GaussianMapper::opacityLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.opacity_lr_;
}
float GaussianMapper::scalingLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.scaling_lr_;
}
float GaussianMapper::rotationLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.rotation_lr_;
}
float GaussianMapper::percentDense()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.percent_dense_;
}
float GaussianMapper::lambdaDssim()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.lambda_dssim_;
}
int GaussianMapper::opacityResetInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.opacity_reset_interval_;
}
float GaussianMapper::densifyGradThreshold()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densify_grad_threshold_;
}
int GaussianMapper::densifyInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densification_interval_;
}
int GaussianMapper::newKeyframeTimesOfUse()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return new_keyframe_times_of_use_;
}
int GaussianMapper::stableNumIterExistence()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return stable_num_iter_existence_;
}
bool GaussianMapper::isKeepingTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return keep_training_;
}
bool GaussianMapper::isdoingGausPyramidTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return do_gaus_pyramid_training_;
}
bool GaussianMapper::isdoingInactiveGeoDensify()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return inactive_geo_densify_;
}

void GaussianMapper::setPositionLearningRateInit(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.position_lr_init_ = lr;
}
void GaussianMapper::setFeatureLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.feature_lr_ = lr;
}
void GaussianMapper::setOpacityLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.opacity_lr_ = lr;
}
void GaussianMapper::setScalingLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.scaling_lr_ = lr;
}
void GaussianMapper::setRotationLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.rotation_lr_ = lr;
}
void GaussianMapper::setPercentDense(const float percent_dense)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.percent_dense_ = percent_dense;
    gaussians_->setPercentDense(percent_dense);
}
void GaussianMapper::setLambdaDssim(const float lambda_dssim)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.lambda_dssim_ = lambda_dssim;
}
void GaussianMapper::setOpacityResetInterval(const int interval)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.opacity_reset_interval_ = interval;
}
void GaussianMapper::setDensifyGradThreshold(const float th)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.densify_grad_threshold_ = th;
}
void GaussianMapper::setDensifyInterval(const int interval)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.densification_interval_ = interval;
}
void GaussianMapper::setNewKeyframeTimesOfUse(const int times)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    new_keyframe_times_of_use_ = times;
}
void GaussianMapper::setStableNumIterExistence(const int niter)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    stable_num_iter_existence_ = niter;
}
void GaussianMapper::setKeepTraining(const bool keep)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    keep_training_ = keep;
}
void GaussianMapper::setDoGausPyramidTraining(const bool gaus_pyramid)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    do_gaus_pyramid_training_ = gaus_pyramid;
}
void GaussianMapper::setDoInactiveGeoDensify(const bool inactive_geo_densify)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    inactive_geo_densify_ = inactive_geo_densify;
}

VariableParameters GaussianMapper::getVaribleParameters()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    VariableParameters params;
    params.position_lr_init = opt_params_.position_lr_init_;
    params.feature_lr = opt_params_.feature_lr_;
    params.opacity_lr = opt_params_.opacity_lr_;
    params.scaling_lr = opt_params_.scaling_lr_;
    params.rotation_lr = opt_params_.rotation_lr_;
    params.percent_dense = opt_params_.percent_dense_;
    params.lambda_dssim = opt_params_.lambda_dssim_;
    params.opacity_reset_interval = opt_params_.opacity_reset_interval_;
    params.densify_grad_th = opt_params_.densify_grad_threshold_;
    params.densify_interval = opt_params_.densification_interval_;
    params.new_kf_times_of_use = new_keyframe_times_of_use_;
    params.stable_num_iter_existence = stable_num_iter_existence_;
    params.keep_training = keep_training_;
    params.do_gaus_pyramid_training = do_gaus_pyramid_training_;
    params.do_inactive_geo_densify = inactive_geo_densify_;
    return params;
}

void GaussianMapper::setVaribleParameters(const VariableParameters &params)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.position_lr_init_ = params.position_lr_init;
    opt_params_.feature_lr_ = params.feature_lr;
    opt_params_.opacity_lr_ = params.opacity_lr;
    opt_params_.scaling_lr_ = params.scaling_lr;
    opt_params_.rotation_lr_ = params.rotation_lr;
    opt_params_.percent_dense_ = params.percent_dense;
    gaussians_->setPercentDense(params.percent_dense);
    opt_params_.lambda_dssim_ = params.lambda_dssim;
    opt_params_.opacity_reset_interval_ = params.opacity_reset_interval;
    opt_params_.densify_grad_threshold_ = params.densify_grad_th;
    opt_params_.densification_interval_ = params.densify_interval;
    new_keyframe_times_of_use_ = params.new_kf_times_of_use;
    stable_num_iter_existence_ = params.stable_num_iter_existence;
    keep_training_ = params.keep_training;
    do_gaus_pyramid_training_ = params.do_gaus_pyramid_training;
    inactive_geo_densify_ = params.do_inactive_geo_densify;
}

void GaussianMapper::loadPly(std::filesystem::path ply_path, std::filesystem::path camera_path)
{
    this->gaussians_->loadPly(ply_path);
    loadCamera(camera_path);

    // Ready
    this->initial_mapped_ = true;
}

void GaussianMapper::loadCompact(std::filesystem::path compact_path, std::filesystem::path camera_path)
{
    if (compact_path.filename() == "metadata.json")
        compact_path = compact_path.parent_path();
    const auto metadata_path = compact_path / "metadata.json";
    if (!std::filesystem::exists(metadata_path))
        throw std::runtime_error("Cannot find compact metadata at " + metadata_path.string());

    const auto format = compactMetadataFormat(metadata_path);
    if (format == "phase2_residual_field_compact") {
        auto decoded = phase2_residual_field::loadPhase2Compact(compact_path, device_type_);
        this->gaussians_->loadCompactDecoded(decoded);
    }
    else {
        this->gaussians_->loadCompact(compact_path);
    }
    loadCamera(camera_path);

    // Ready
    this->initial_mapped_ = true;
}

void GaussianMapper::loadResult(std::filesystem::path result_path, std::filesystem::path camera_path)
{
    if (looksLikeCompactResultPath(result_path))
        loadCompact(result_path, camera_path);
    else
        loadPly(result_path, camera_path);
}

void GaussianMapper::loadCamera(std::filesystem::path camera_path)
{
    if (camera_path.empty())
        return;

    // Camera
    if (std::filesystem::exists(camera_path)) {
        scene_->cameras_.clear();
        undistort_mask_.clear();
        viewer_main_undistort_mask_.clear();
        viewer_sub_undistort_mask_.clear();
        viewer_camera_id_set_ = false;

        cv::FileStorage camera_file(camera_path.string().c_str(), cv::FileStorage::READ);
        if(!camera_file.isOpened())
            throw std::runtime_error("[Gaussian Mapper]Failed to open settings file at: " + camera_path.string());

        Camera camera;
        camera.camera_id_ = 0;
        camera.width_ = camera_file["Camera.w"].operator int();
        camera.height_ = camera_file["Camera.h"].operator int();

        std::string camera_type = camera_file["Camera.type"].string();
        if (camera_type == "Pinhole") {
            camera.setModelId(Camera::CameraModelType::PINHOLE);

            float fx = camera_file["Camera.fx"].operator float();
            float fy = camera_file["Camera.fy"].operator float();
            float cx = camera_file["Camera.cx"].operator float();
            float cy = camera_file["Camera.cy"].operator float();

            float k1 = camera_file["Camera.k1"].operator float();
            float k2 = camera_file["Camera.k2"].operator float();
            float p1 = camera_file["Camera.p1"].operator float();
            float p2 = camera_file["Camera.p2"].operator float();
            float k3 = camera_file["Camera.k3"].operator float();

            cv::Mat K = (
                cv::Mat_<float>(3, 3)
                    << fx, 0.f, cx,
                        0.f, fy, cy,
                        0.f, 0.f, 1.f
            );

            camera.params_[0] = fx;
            camera.params_[1] = fy;
            camera.params_[2] = cx;
            camera.params_[3] = cy;

            std::vector<float> dist_coeff = {k1, k2, p1, p2, k3};
            camera.dist_coeff_ = cv::Mat(5, 1, CV_32F, dist_coeff.data());
            camera.initUndistortRectifyMapAndMask(K, cv::Size(camera.width_, camera.height_), K, false);

            undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    camera.undistort_mask, device_type_);

            cv::Mat viewer_main_undistort_mask;
            int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
            int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
            cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                       cv::Size(viewer_image_width_main_, viewer_image_height_main_));
            viewer_main_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_main_undistort_mask, device_type_);


            if (!camera_file["Camera.depth_scale"].empty())
                camera.depth_scale_ = camera_file["Camera.depth_scale"].operator float();
        }
        else if (camera_type == "Lonlat") {
            camera.setModelId(Camera::CameraModelType::LONLAT);

            cv::Mat K = (
                cv::Mat_<float>(3, 3)
                    << camera.width_, 0.f, camera.width_ / 2.0f,
                        0.f, camera.height_, camera.height_ / 2.0f,
                        0.f, 0.f, 1.f
            );
            camera.initUndistortRectifyMapAndMask(K, cv::Size(camera.width_, camera.height_), K, false);

            undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    camera.undistort_mask, device_type_);

            cv::Mat viewer_main_undistort_mask;
            int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
            int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
            cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                       cv::Size(viewer_image_width_main_, viewer_image_height_main_));
            viewer_main_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_main_undistort_mask, device_type_);
        }
        else {
            throw std::runtime_error("[Gaussian Mapper]Unsupported camera model: " + camera_path.string());
        }

        if (!viewer_camera_id_set_) {
            viewer_camera_id_ = camera.camera_id_;
            viewer_camera_id_set_ = true;
        }
        this->scene_->addCamera(camera);
    }
}

void GaussianMapper::keyframesFromJson(std::filesystem::path json_path)
{
    std::ifstream in_stream;
    in_stream.open(json_path);
    if (!in_stream.is_open())
        throw std::runtime_error("Cannot open json file at " + json_path.string());

    Json::Value json_root;
    Json::CharReaderBuilder builder;
    JSONCPP_STRING errs;
    if (!Json::parseFromStream(builder, in_stream, &json_root, &errs))
        throw std::runtime_error(errs);

    for (const auto& json_kf : json_root) {
        // Read json data
        std::size_t fid = json_kf["id"].asUInt();
        std::string img_filename = json_kf["img_name"].asString();
        int width = json_kf["width"].asUInt();
        int height = json_kf["height"].asUInt();

        float fy = json_kf["fy"].asFloat();
        float fx = json_kf["fx"].asFloat();

        Eigen::Vector3f pos;
        pos.x() = json_kf["position"][0].asFloat();
        pos.y() = json_kf["position"][1].asFloat();
        pos.z() = json_kf["position"][2].asFloat();

        Eigen::Matrix3f rot;
        rot(0, 0) = json_kf["rotation"][0][0].asFloat();
        rot(0, 1) = json_kf["rotation"][0][1].asFloat();
        rot(0, 2) = json_kf["rotation"][0][2].asFloat();
        rot(1, 0) = json_kf["rotation"][1][0].asFloat();
        rot(1, 1) = json_kf["rotation"][1][1].asFloat();
        rot(1, 2) = json_kf["rotation"][1][2].asFloat();
        rot(2, 0) = json_kf["rotation"][2][0].asFloat();
        rot(2, 1) = json_kf["rotation"][2][1].asFloat();
        rot(2, 2) = json_kf["rotation"][2][2].asFloat();

        // Create the corresponding new keyframe and add it to the scene
        std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>(fid);
        pkf->img_filename_ = img_filename; // TODO: read the image if exists, warn otherwise
        pkf->zfar_ = z_far_;
        pkf->znear_ = z_near_;
        // Pose
        pkf->setPose(
            Eigen::Quaterniond(rot.cast<double>()),
            pos.cast<double>());
        try {
            // Camera
            Camera& camera = scene_->cameras_.at(viewer_camera_id_);
            assert(width == camera.width_ && height == camera.height_);
            pkf->setCameraParams(camera);
            // Transformations
            pkf->computeTransformTensors();
        }
        catch (std::out_of_range) {
            throw std::runtime_error("[GaussianMapper::keyframesFromJson]KeyFrame Camera not found!");
        }
    }
}

void GaussianMapper::renderAndEstimateDepth(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    // TODO: need the ground truth depth image
}
