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

#include "include/gaussian_renderer.h"

/**
 * @brief 
 * 
 * @return std::tuple<render, viewspace_points, visibility_filter, radii>, which are all `torch::Tensor`
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GaussianRenderer::render(
    std::shared_ptr<GaussianKeyframe> viewpoint_camera,
    int image_height,
    int image_width,
    std::shared_ptr<GaussianModel> pc,
    GaussianPipelineParams& pipe,
    torch::Tensor& bg_color,
    torch::Tensor& override_color,
    bool render_depth,
    float scaling_modifier,
    bool use_override_color)
{
    /* Render the scene. 

       Background tensor (bg_color) must be on GPU!
     */

    // Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    auto screenspace_points = torch::zeros(
        {pc->getXYZ().size(0), 4},
        torch::TensorOptions().dtype(pc->getXYZ().dtype()).requires_grad(true).device(torch::kCUDA));
    try {
        screenspace_points.retain_grad();
    }
    catch (const std::exception& e) {
        ; // pass
    }

    // Set up rasterization configuration
    float tanfovx = std::tan(viewpoint_camera->FoVx_ * 0.5f);
    float tanfovy = std::tan(viewpoint_camera->FoVy_ * 0.5f);

    GaussianRasterizationSettings raster_settings(
        image_height,
        image_width,
        tanfovx,
        tanfovy,
        bg_color,
        scaling_modifier,
        viewpoint_camera->world_view_transform_,
        viewpoint_camera->full_proj_transform_,
        viewpoint_camera->RwcT_,
        pc->active_sh_degree_,
        viewpoint_camera->camera_center_,
        false,
        Camera::CameraModelType::PINHOLE,
        render_depth
    );

    GaussianRasterizer rasterizer(raster_settings);

    auto means3D = pc->getXYZ();
    auto means2D = screenspace_points;
    auto opacity = pc->getOpacityActivation();

    /* If precomputed 3d covariance is provided, use it. If not, then it will be computed from
       scaling / rotation by the rasterizer. 
     */
    bool has_scales = false,
         has_rotations = false,
         has_cov3D_precomp = false;
    torch::Tensor scales,
                  rotations,
                  cov3D_precomp;
    if (pipe.compute_cov3D_) {
        cov3D_precomp = pc->getCovarianceActivation();
        has_cov3D_precomp = true;
    }
    else {
        scales = pc->getScalingActivation();
        rotations = pc->getRotationActivation();
        has_scales = true;
        has_rotations = true;
    }

    /* If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
       from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
     */
    bool has_shs = false,
         has_color_precomp = false;
    torch::Tensor shs,
                  colors_precomp;
    if (use_override_color) {
        colors_precomp = override_color;
        has_color_precomp = true;
    }
    else {
        if (pipe.convert_SHs_) {
            int max_sh_degree = pc->max_sh_degree_ + 1;
            torch::Tensor shs_view = pc->getFeatures().transpose(1, 2).view({-1, 3, max_sh_degree * max_sh_degree});
            torch::Tensor dir_pp = (pc->getXYZ() - viewpoint_camera->camera_center_.repeat({pc->getFeatures().size(0), 1}));
            auto dir_pp_normalized = dir_pp / torch::frobenius_norm(dir_pp, /*dim=*/{1}, /*keepdim=*/true);
            auto sh2rgb = sh_utils::eval_sh(pc->active_sh_degree_, shs_view, dir_pp_normalized);
            colors_precomp = torch::clamp_min(sh2rgb + 0.5, 0.0);
            has_color_precomp = true;
        }
        else {
            shs = pc->getFeatures();
            has_shs = true;
        }
    }

    // Rasterize visible Gaussians to image, obtain their radii (on screen). 
    auto rasterizer_result = rasterizer.forward(
        means3D,
        means2D,
        opacity,
        has_shs,
        has_color_precomp,
        has_scales,
        has_rotations,
        has_cov3D_precomp,
        shs,
        colors_precomp,
        scales,
        rotations,
        cov3D_precomp
    );
    auto rendered_image = std::get<0>(rasterizer_result);
    auto radii = std::get<1>(rasterizer_result);

    /* Those Gaussians that were frustum culled or had a radius of 0 were not visible.
       They will be excluded from value updates used in the splitting criteria.
     */
    return std::make_tuple(
        rendered_image,     /*render*/
        screenspace_points, /*viewspace_points*/
        radii > 0,          /*visibility_filter*/
        radii               /*radii*/
    );
}

/**
 * @brief 
 * 
 * @return std::tuple<render, viewspace_points, visibility_filter, radii>, which are all `torch::Tensor`
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GaussianRenderer::renderLonlat(
    std::shared_ptr<GaussianKeyframe> viewpoint_camera,
    int image_height,
    int image_width,
    std::shared_ptr<GaussianModel> pc,
    GaussianPipelineParams& pipe,
    torch::Tensor& bg_color,
    torch::Tensor& override_color,
    float scaling_modifier,
    bool use_override_color)
{
    /* Render the scene. 

       Background tensor (bg_color) must be on GPU!
     */

    // Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    auto screenspace_points = torch::zeros(
        {pc->getXYZ().size(0), 4},
        torch::TensorOptions().dtype(pc->getXYZ().dtype()).requires_grad(true).device(torch::kCUDA));
    try {
        screenspace_points.retain_grad();
    }
    catch (const std::exception& e) {
        ; // pass
    }

    // Set up rasterization configuration
    GaussianRasterizationSettings raster_settings(
        image_height,
        image_width,
        0.0f,
        0.0f,
        bg_color,
        scaling_modifier,
        viewpoint_camera->world_view_transform_,
        viewpoint_camera->full_proj_transform_,
        viewpoint_camera->RwcT_,
        pc->active_sh_degree_,
        viewpoint_camera->camera_center_,
        false,
        Camera::CameraModelType::LONLAT
    );

    GaussianRasterizer rasterizer(raster_settings);

    auto means3D = pc->getXYZ();
    auto means2D = screenspace_points;
    auto opacity = pc->getOpacityActivation();

    /* If precomputed 3d covariance is provided, use it. If not, then it will be computed from
       scaling / rotation by the rasterizer. 
     */
    bool has_scales = false,
         has_rotations = false,
         has_cov3D_precomp = false;
    torch::Tensor scales,
                  rotations,
                  cov3D_precomp;
    if (pipe.compute_cov3D_) {
        cov3D_precomp = pc->getCovarianceActivation();
        has_cov3D_precomp = true;
    }
    else {
        scales = pc->getScalingActivation();
        rotations = pc->getRotationActivation();
        has_scales = true;
        has_rotations = true;
    }

    /* If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
       from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
     */
    bool has_shs = false,
         has_color_precomp = false;
    torch::Tensor shs,
                  colors_precomp;
    if (use_override_color) {
        colors_precomp = override_color;
        has_color_precomp = true;
    }
    else {
        if (pipe.convert_SHs_) {
            int max_sh_degree = pc->max_sh_degree_ + 1;
            torch::Tensor shs_view = pc->getFeatures().transpose(1, 2).view({-1, 3, max_sh_degree * max_sh_degree});
            torch::Tensor dir_pp = (pc->getXYZ() - viewpoint_camera->camera_center_.repeat({pc->getFeatures().size(0), 1}));
            auto dir_pp_normalized = dir_pp / torch::frobenius_norm(dir_pp, /*dim=*/{1}, /*keepdim=*/true);
            auto sh2rgb = sh_utils::eval_sh(pc->active_sh_degree_, shs_view, dir_pp_normalized);
            colors_precomp = torch::clamp_min(sh2rgb + 0.5, 0.0);
            has_color_precomp = true;
        }
        else {
            shs = pc->getFeatures();
            has_shs = true;
        }
    }

    // Rasterize visible Gaussians to image, obtain their radii (on screen). 
    auto rasterizer_result = rasterizer.forward(
        means3D,
        means2D,
        opacity,
        has_shs,
        has_color_precomp,
        has_scales,
        has_rotations,
        has_cov3D_precomp,
        shs,
        colors_precomp,
        scales,
        rotations,
        cov3D_precomp
    );
    auto rendered_image = std::get<0>(rasterizer_result);
    auto radii = std::get<1>(rasterizer_result);

    /* Those Gaussians that were frustum culled or had a radius of 0 were not visible.
       They will be excluded from value updates used in the splitting criteria.
     */
    return std::make_tuple(
        rendered_image,     /*render*/
        screenspace_points, /*viewspace_points*/
        radii > 0,          /*visibility_filter*/
        radii               /*radii*/
    );
}
