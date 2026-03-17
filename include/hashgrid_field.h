/**
 * This file is part of OmniGS
 */

#pragma once

#include <torch/torch.h>

struct HashGridEncoderOptions
{
    int num_levels = 8;
    int features_per_level = 2;
    int log2_hashmap_size = 18;
    int base_resolution = 16;
    float per_level_scale = 1.5f;
};

class HashGridEncoderImpl : public torch::nn::Module
{
public:
    explicit HashGridEncoderImpl(const HashGridEncoderOptions& options);

    torch::Tensor forward(const torch::Tensor& xyz_normalized);

    int outputDim() const { return num_levels_ * features_per_level_; }

private:
    torch::Tensor hashIndices(const torch::Tensor& coords) const;

private:
    int num_levels_ = 0;
    int features_per_level_ = 0;
    int table_size_ = 0;
    std::vector<int> resolutions_;
    std::vector<torch::Tensor> tables_;
};

TORCH_MODULE(HashGridEncoder);
