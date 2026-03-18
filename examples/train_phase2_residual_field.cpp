#include <filesystem>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "include/phase2_residual_field.h"

namespace
{

template <typename T>
void readIfPresent(const cv::FileStorage& settings, const std::string& key, T& value)
{
    if (!settings[key].empty())
        value = settings[key].operator T();
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc != 4)
        {
            std::cerr << std::endl
                      << "Usage: " << argv[0]
                      << " path_to_gaussian_mapping_settings"    /*1*/
                      << " path_to_phase2_frozen_package"        /*2*/
                      << " path_to_output_directory/"            /*3*/
                      << std::endl;
            return 1;
        }

        torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        if (device_type == torch::kCUDA)
            std::cout << "CUDA available! Training Phase 2 residual field on GPU." << std::endl;
        else
            std::cout << "Training Phase 2 residual field on CPU." << std::endl;

        std::filesystem::path gaussian_cfg_path(argv[1]);
        std::filesystem::path input_phase2_dir(argv[2]);
        std::filesystem::path output_dir(argv[3]);

        cv::FileStorage settings(gaussian_cfg_path.string().c_str(), cv::FileStorage::READ);
        if (!settings.isOpened()) {
            std::cerr << "[Phase2ResidualField] Failed to open settings file at: " << gaussian_cfg_path << std::endl;
            return 1;
        }

        Phase2ResidualFieldTrainOptions options;
        readIfPresent(settings, "Phase2Field.num_fourier_frequencies", options.num_fourier_frequencies);
        if (!settings["Phase2Field.use_hashgrid_encoder"].empty())
            options.use_hashgrid_encoder = (settings["Phase2Field.use_hashgrid_encoder"].operator int()) != 0;
        readIfPresent(settings, "Phase2Field.hashgrid_num_levels", options.hashgrid_num_levels);
        readIfPresent(settings, "Phase2Field.hashgrid_features_per_level", options.hashgrid_features_per_level);
        readIfPresent(settings, "Phase2Field.hashgrid_log2_hashmap_size", options.hashgrid_log2_hashmap_size);
        readIfPresent(settings, "Phase2Field.hashgrid_base_resolution", options.hashgrid_base_resolution);
        readIfPresent(settings, "Phase2Field.hashgrid_per_level_scale", options.hashgrid_per_level_scale);
        readIfPresent(settings, "Phase2Field.hidden_dim", options.hidden_dim);
        readIfPresent(settings, "Phase2Field.num_hidden_layers", options.num_hidden_layers);
        readIfPresent(settings, "Phase2Field.batch_size", options.batch_size);
        readIfPresent(settings, "Phase2Field.max_steps", options.max_steps);
        readIfPresent(settings, "Phase2Field.log_interval", options.log_interval);
        readIfPresent(settings, "Phase2Field.eval_interval", options.eval_interval);
        readIfPresent(settings, "Phase2Field.learning_rate", options.learning_rate);
        readIfPresent(settings, "Phase2Field.weight_decay", options.weight_decay);
        if (!settings["Phase2Field.include_features_dc"].empty())
            options.include_features_dc = (settings["Phase2Field.include_features_dc"].operator int()) != 0;
        if (!settings["Phase2Field.include_opacity"].empty())
            options.include_opacity = (settings["Phase2Field.include_opacity"].operator int()) != 0;
        if (!settings["Phase2Field.include_scaling"].empty())
            options.include_scaling = (settings["Phase2Field.include_scaling"].operator int()) != 0;
        if (!settings["Phase2Field.include_rotation"].empty())
            options.include_rotation = (settings["Phase2Field.include_rotation"].operator int()) != 0;
        if (!settings["Phase2Field.predict_opacity"].empty())
            options.predict_opacity = (settings["Phase2Field.predict_opacity"].operator int()) != 0;
        if (!settings["Phase2Field.predict_scaling"].empty())
            options.predict_scaling = (settings["Phase2Field.predict_scaling"].operator int()) != 0;
        if (!settings["Phase2Field.predict_rotation"].empty())
            options.predict_rotation = (settings["Phase2Field.predict_rotation"].operator int()) != 0;
        readIfPresent(settings, "Phase2Field.block_embedding_dim", options.block_embedding_dim);
        if (!settings["Phase2Field.hybrid_hard_only"].empty())
            options.hybrid_hard_only = (settings["Phase2Field.hybrid_hard_only"].operator int()) != 0;
        if (!settings["Phase2Field.hybrid_override_rest_only"].empty())
            options.hybrid_override_rest_only = (settings["Phase2Field.hybrid_override_rest_only"].operator int()) != 0;
        if (!settings["Phase2Field.hybrid_easy_export_sh_drop"].empty())
            options.hybrid_easy_export_sh_drop = (settings["Phase2Field.hybrid_easy_export_sh_drop"].operator int()) != 0;
        if (!settings["Phase2Field.hybrid_easy_export_sh_preserve_blocks"].empty())
            options.hybrid_easy_export_sh_preserve_blocks = (settings["Phase2Field.hybrid_easy_export_sh_preserve_blocks"].operator int()) != 0;
        readIfPresent(settings, "Phase2Field.hybrid_easy_export_sh_energy_keep_ratio", options.hybrid_easy_export_sh_energy_keep_ratio);
        readIfPresent(settings, "Phase2Field.hybrid_easy_export_sh_min_opacity", options.hybrid_easy_export_sh_min_opacity);
        readIfPresent(settings, "Phase2Field.hybrid_easy_export_sh_min_level", options.hybrid_easy_export_sh_min_level);
        if (!settings["Phase2Field.save_decoded_compact"].empty())
            options.save_decoded_compact = (settings["Phase2Field.save_decoded_compact"].operator int()) != 0;
        if (!settings["Phase2Field.save_phase2_compact"].empty())
            options.save_phase2_compact = (settings["Phase2Field.save_phase2_compact"].operator int()) != 0;
        readIfPresent(settings, "Phase2Field.decoded_xyz_quant_bits", options.decoded_xyz_quant_bits);
        readIfPresent(settings, "Phase2Field.decoded_attribute_quant_bits", options.decoded_attribute_quant_bits);
        readIfPresent(settings, "Phase2Field.decoded_rotation_quant_bits", options.decoded_rotation_quant_bits);
        readIfPresent(settings, "Phase2Field.phase2_compact_opacity_quant_bits", options.phase2_compact_opacity_quant_bits);
        readIfPresent(settings, "Phase2Field.phase2_compact_scaling_quant_bits", options.phase2_compact_scaling_quant_bits);
        if (!settings["Phase2Field.phase2_compact_pack_sh_levels"].empty())
            options.phase2_compact_pack_sh_levels = (settings["Phase2Field.phase2_compact_pack_sh_levels"].operator int()) != 0;
        readIfPresent(settings, "Phase2Field.phase2_compact_fdc_quant_bits", options.phase2_compact_fdc_quant_bits);
        readIfPresent(settings, "Phase2Field.phase2_compact_easy_rest_base_quant_bits", options.phase2_compact_easy_rest_base_quant_bits);
        readIfPresent(settings, "Phase2Field.phase2_compact_easy_rest_scale_quant_bits", options.phase2_compact_easy_rest_scale_quant_bits);
        readIfPresent(settings, "Phase2Field.phase2_compact_easy_rest_int2_rel_mse_threshold", options.phase2_compact_easy_rest_int2_rel_mse_threshold);
        if (!settings["Phase2Field.phase2_compact_use_geometry_codec"].empty())
            options.phase2_compact_use_geometry_codec = (settings["Phase2Field.phase2_compact_use_geometry_codec"].operator int()) != 0;
        readIfPresent(settings, "Phase2Field.phase2_compact_geometry_quant_bits", options.phase2_compact_geometry_quant_bits);
        if (!settings["Phase2Field.phase2_compact_store_field_fp16"].empty())
            options.phase2_compact_store_field_fp16 = (settings["Phase2Field.phase2_compact_store_field_fp16"].operator int()) != 0;
        if (!settings["Phase2Field.phase2_compact_easy_rest_zlib"].empty())
            options.phase2_compact_easy_rest_zlib = (settings["Phase2Field.phase2_compact_easy_rest_zlib"].operator int()) != 0;
        readIfPresent(settings, "Phase2Field.phase2_compact_easy_rest_zlib_level", options.phase2_compact_easy_rest_zlib_level);
        if (!settings["Phase2Field.phase2_compact_quantized_tensor_zlib"].empty())
            options.phase2_compact_quantized_tensor_zlib = (settings["Phase2Field.phase2_compact_quantized_tensor_zlib"].operator int()) != 0;
        readIfPresent(settings, "Phase2Field.phase2_compact_quantized_tensor_zlib_level", options.phase2_compact_quantized_tensor_zlib_level);
        if (!settings["Phase2Field.phase2_compact_geometry_zlib"].empty())
            options.phase2_compact_geometry_zlib = (settings["Phase2Field.phase2_compact_geometry_zlib"].operator int()) != 0;
        readIfPresent(settings, "Phase2Field.phase2_compact_geometry_zlib_level", options.phase2_compact_geometry_zlib_level);
        if (!settings["Phase2Field.phase2_compact_field_zlib"].empty())
            options.phase2_compact_field_zlib = (settings["Phase2Field.phase2_compact_field_zlib"].operator int()) != 0;
        readIfPresent(settings, "Phase2Field.phase2_compact_field_zlib_level", options.phase2_compact_field_zlib_level);
        if (!settings["Phase2Field.phase2_compact_use_xz"].empty())
            options.phase2_compact_use_xz = (settings["Phase2Field.phase2_compact_use_xz"].operator int()) != 0;

        auto frozen = phase2_residual_field::loadFrozenPackage(input_phase2_dir, device_type);
        auto result = phase2_residual_field::trainResidualField(frozen, options, output_dir);

        std::cout << "[Phase2ResidualField] best_eval_mse=" << result.best_loss
                  << " final_eval_mse=" << result.final_eval_loss
                  << " trained_steps=" << result.trained_steps
                  << std::endl;

        return 0;
    }
    catch (const c10::Error& err) {
        std::cerr << "[Phase2ResidualField] C10 error: " << err.what() << std::endl;
        return 1;
    }
    catch (const std::exception& err) {
        std::cerr << "[Phase2ResidualField] Exception: " << err.what() << std::endl;
        return 1;
    }
}
