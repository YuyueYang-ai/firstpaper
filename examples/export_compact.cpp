#include <filesystem>
#include <iostream>
#include <memory>

#include <torch/torch.h>

#include "include/gaussian_mapper.h"

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << std::endl
                  << "Usage: " << argv[0]
                  << " path_to_gaussian_mapping_settings"    /*1*/
                  << " path_to_input_model"                  /*2: ply or compact dir */
                  << " path_to_output_directory/"            /*3*/
                  << std::endl;
        return 1;
    }

    torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    if (device_type == torch::kCUDA)
        std::cout << "CUDA available! Exporting on GPU." << std::endl;
    else
        std::cout << "Exporting on CPU." << std::endl;

    std::filesystem::path gaussian_cfg_path(argv[1]);
    std::filesystem::path input_model_path(argv[2]);
    std::filesystem::path output_dir(argv[3]);

    std::shared_ptr<GaussianMapper> mapper =
        std::make_shared<GaussianMapper>(gaussian_cfg_path, std::filesystem::path(), 0, device_type);
    mapper->loadResult(input_model_path);
    mapper->exportCompact(output_dir);

    return 0;
}
