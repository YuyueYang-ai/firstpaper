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

#include <math.h>
#include <torch/torch.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "include/rasterize_points.h"
#include <fstream>
#include <string>
#include <functional>

/**
 * Tensor一维化
 */
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const int camera_type,
	const bool render_depth)
{
	if (means3D.ndimension() != 2 || means3D.size(1) != 3)
	{
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}

	const int P = means3D.size(0); // 模型点数
	const int H = image_height;    // 图像高（像素数）
	const int W = image_width;     // 图像宽（像素数）

	auto int_opts = means3D.options().dtype(torch::kInt32); // not used
	auto float_opts = means3D.options().dtype(torch::kFloat32);
	// 创建输出Tensor
	torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
	torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
	// 创建过程buffer及其resize函数
	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);       // 模型3D点云的点
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer); // point list和key（用于排序，为了alpha-blending)
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);         // 渲染结果2D图像的点

	int rendered = 0;
	if (P != 0)
	{
		int M = 0; // Gaussian Model用SH（球谐波）表示颜色（f_dc和f_rest），M是dc和rest谐波之和(1+15)
		if (sh.size(0) != 0)
		{
			M = sh.size(1);
		}

		if (camera_type == 1) // PINHOLE
		{
			// CUDA实现在rasterizer_impl.cu；torch和CUDA实现分开在两个文件中，使内层没有torch
			rendered = CudaRasterizer::Rasterizer::forward(
				geomFunc,
				binningFunc,
				imgFunc,
				P, degree, M,
				background.contiguous().data_ptr<float>(),
				W, H,
				means3D.contiguous().data_ptr<float>(),
				sh.contiguous().data_ptr<float>(),
				colors.contiguous().data_ptr<float>(),
				opacity.contiguous().data_ptr<float>(),
				scales.contiguous().data_ptr<float>(),
				scale_modifier,
				rotations.contiguous().data_ptr<float>(),
				cov3D_precomp.contiguous().data_ptr<float>(),
				viewmatrix.contiguous().data_ptr<float>(),
				projmatrix.contiguous().data_ptr<float>(),
				campos.contiguous().data_ptr<float>(),
				tan_fovx,
				tan_fovy,
				prefiltered,
				out_color.contiguous().data_ptr<float>(),
				radii.contiguous().data_ptr<int>(),
				render_depth);
		}
		else if (camera_type == 3) // LONLAT
		{
// torch::Tensor radii_x = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
			rendered = CudaRasterizer::LonlatRasterizer::forward(
				geomFunc,
				binningFunc,
				imgFunc,
				P, degree, M,
				background.contiguous().data_ptr<float>(),
				W, H,
				means3D.contiguous().data_ptr<float>(),
				sh.contiguous().data_ptr<float>(),
				colors.contiguous().data_ptr<float>(),
				opacity.contiguous().data_ptr<float>(),
				scales.contiguous().data_ptr<float>(),
				scale_modifier,
				rotations.contiguous().data_ptr<float>(),
				cov3D_precomp.contiguous().data_ptr<float>(),
				viewmatrix.contiguous().data_ptr<float>(),
				campos.contiguous().data_ptr<float>(),
				prefiltered,
				out_color.contiguous().data_ptr<float>(),
				radii.contiguous().data_ptr<int>());
// radii_x.contiguous().data_ptr<int>());
		}
		else
		{
			throw std::runtime_error("[CudaRasterizer]Invalid camera_type");
		}
	}
	return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int camera_type) 
{
	const int P = means3D.size(0);
	const int H = dL_dout_color.size(1);
	const int W = dL_dout_color.size(2);

	int M = 0;
	if(sh.size(0) != 0)
	{	
		M = sh.size(1);
	}

	torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dmeans2D = torch::zeros({P, 4}, means3D.options());
	torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
	torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
	torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
	torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
	torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

	if(P != 0)
	{
		if (camera_type == 1) // PINHOLE
		{
			CudaRasterizer::Rasterizer::backward(P, degree, M, R,
				background.contiguous().data_ptr<float>(),
				W, H, 
				means3D.contiguous().data_ptr<float>(),
				sh.contiguous().data_ptr<float>(),
				colors.contiguous().data_ptr<float>(),
				scales.data_ptr<float>(),
				scale_modifier,
				rotations.data_ptr<float>(),
				cov3D_precomp.contiguous().data_ptr<float>(),
				viewmatrix.contiguous().data_ptr<float>(),
				projmatrix.contiguous().data_ptr<float>(),
				campos.contiguous().data_ptr<float>(),
				tan_fovx,
				tan_fovy,
				radii.contiguous().data_ptr<int>(),
				reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
				reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
				reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
				dL_dout_color.contiguous().data_ptr<float>(),
				dL_dmeans2D.contiguous().data_ptr<float>(),
				dL_dconic.contiguous().data_ptr<float>(),  
				dL_dopacity.contiguous().data_ptr<float>(),
				dL_dcolors.contiguous().data_ptr<float>(),
				dL_dmeans3D.contiguous().data_ptr<float>(),
				dL_dcov3D.contiguous().data_ptr<float>(),
				dL_dsh.contiguous().data_ptr<float>(),
				dL_dscales.contiguous().data_ptr<float>(),
				dL_drotations.contiguous().data_ptr<float>());
		}
		else if (camera_type == 3) // LONLAT
		{
			torch::Tensor dpx_dt = torch::zeros({P, 3}, means3D.options());
			torch::Tensor dpy_dt = torch::zeros({P, 3}, means3D.options());

			CudaRasterizer::LonlatRasterizer::backward(P, degree, M, R,
				background.contiguous().data_ptr<float>(),
				W, H, 
				means3D.contiguous().data_ptr<float>(),
				sh.contiguous().data_ptr<float>(),
				colors.contiguous().data_ptr<float>(),
				scales.data_ptr<float>(),
				scale_modifier,
				rotations.data_ptr<float>(),
				cov3D_precomp.contiguous().data_ptr<float>(),
				viewmatrix.contiguous().data_ptr<float>(),
				campos.contiguous().data_ptr<float>(),
				radii.contiguous().data_ptr<int>(),
				reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
				reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
				reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
				dL_dout_color.contiguous().data_ptr<float>(),
				dL_dmeans2D.contiguous().data_ptr<float>(),
				dL_dconic.contiguous().data_ptr<float>(),  
				dL_dopacity.contiguous().data_ptr<float>(),
				dL_dcolors.contiguous().data_ptr<float>(),
				dL_dmeans3D.contiguous().data_ptr<float>(),
				dL_dcov3D.contiguous().data_ptr<float>(),
				dL_dsh.contiguous().data_ptr<float>(),
				dL_dscales.contiguous().data_ptr<float>(),
				dL_drotations.contiguous().data_ptr<float>(),
				dpx_dt.contiguous().data_ptr<float>(),
				dpy_dt.contiguous().data_ptr<float>());
		}
		else
		{
			throw std::runtime_error("[CudaRasterizer]Invalid camera_type");
		}
	}

	return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
	torch::Tensor& means3D,
	torch::Tensor& viewmatrix,
	torch::Tensor& projmatrix,
	const int camera_type)
{ 
	const int P = means3D.size(0);

	torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

	if(P != 0)
	{
		if (camera_type == 1) // PINHOLE
		{
			CudaRasterizer::Rasterizer::markVisible(P,
				means3D.contiguous().data_ptr<float>(),
				viewmatrix.contiguous().data_ptr<float>(),
				projmatrix.contiguous().data_ptr<float>(),
				present.contiguous().data_ptr<bool>());
		}
		else if (camera_type == 3) // LONLAT
		{
			CudaRasterizer::LonlatRasterizer::markVisible(P,
				present.contiguous().data_ptr<bool>());
		}
		else
		{
			throw std::runtime_error("[CudaRasterizer]Invalid camera_type");
		}
	}

	return present;
}
