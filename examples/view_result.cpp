#include <filesystem>
#include <iostream>
#include <memory>

#include <torch/torch.h>

#include "include/gaussian_mapper.h"
#include "viewer/imgui_viewer.h"

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << std::endl
                  << "Usage: " << argv[0]
                  << " path_to_gaussian_mapping_settings"    /*1*/
                  << " path_to_camera_parameters"            /*2*/
                  << " path_to_result_model"                 /*3: ply or compact dir */
                  << std::endl;
        return 1;
    }

    // Device
    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA available! Viewing on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "Viewing on CPU." << std::endl;
        device_type = torch::kCPU;
    }

    // Create GaussianMapper
    std::filesystem::path gaussian_cfg_path(argv[1]);
    std::filesystem::path camera_path(argv[2]);
    std::filesystem::path result_model_path(argv[3]);
    std::shared_ptr<GaussianMapper> pGausMapper =
        std::make_shared<GaussianMapper>(gaussian_cfg_path, std::filesystem::path(), 0, device_type);
    pGausMapper->loadResult(result_model_path, camera_path);

    // Create Gaussian Viewer
    std::thread viewer_thd;
    std::shared_ptr<ImGuiViewer> pViewer;
    pViewer = std::make_shared<ImGuiViewer>(pGausMapper, false);
    pViewer->run();

    return 0;
}
