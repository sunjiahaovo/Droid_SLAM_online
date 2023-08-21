#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <cmath>

#define THREADS 256
#define NUM_BLOCKS(batch_size) ((batch_size + THREADS - 1) / THREADS)

__global__ void integrate_kernel(
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> intrinsic,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> pose,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> depth,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> volume_origin,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> voxel_size,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> img_size,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> volume, 
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> weight){
    const int voxel_ptx = threadIdx.x + blockDim.x * blockIdx.x;
    const int voxel_pty = threadIdx.y + blockDim.y * blockIdx.y;
    const int voxel_ptz = threadIdx.z + blockDim.z * blockIdx.z;

    float voxel_base_x = volume_origin[0] + voxel_size[0] * voxel_ptx;
    float voxel_base_y = volume_origin[1] + voxel_size[1] * voxel_pty;
    float voxel_base_z = volume_origin[2] + voxel_size[2] * voxel_ptz;

    float voxel_cam_x = voxel_base_x * pose[0][0] + voxel_base_y * pose[0][1] + voxel_base_z * pose[0][2] + pose[0][3];
    float voxel_cam_y = voxel_base_y * pose[1][0] + voxel_base_y * pose[1][1] + voxel_base_z * pose[1][2] + pose[1][3];
    float voxel_cam_z = voxel_base_y * pose[2][0] + voxel_base_y * pose[2][1] + voxel_base_z * pose[2][2] + pose[2][3];

    if(voxel_cam_z <= 0){
        return;
    }

    
    float fx = intrinsic[0];
    float fy = intrinsic[1];
    float cx = intrinsic[2];
    float cy = intrinsic[3];

    int pix_x = roundf(voxel_cam_x / voxel_cam_z * fx + cx);
    int pix_y = roundf(voxel_cam_y / voxel_cam_z * fy + cy);

    
    float img_h = img_size[0];
    float img_w = img_size[1];

    if(pix_x < 0 || pix_x >= img_w || pix_y < 0 || pix_y >= img_h){
        return;
    }

    float depth_val = depth[pix_x][pix_y][0];

    if(depth_val <=0 || depth_val > 6){
        return;
    }

    float diff = depth_val - voxel_cam_z;

    const float trunc_margin = fmax(voxel_size[0], fmax(voxel_size[1], voxel_size[2]));

    if(diff < - trunc_margin){
        return;
    }

    float dist = fmin(1.0f, diff / trunc_margin);
    float weight_old = weight[voxel_ptx][voxel_pty][voxel_ptz];
    float weight_new = weight_old + 1.0f;
    weight[voxel_ptx][voxel_pty][voxel_ptz] = weight_new;
    volume[voxel_ptx][voxel_pty][voxel_ptz] = (volume[voxel_ptx][voxel_pty][voxel_ptz] * weight_old + dist) / weight_new;
    return;
}


void integrate_cuda(
    torch::Tensor intrinsic,
    torch::Tensor pose,
    torch::Tensor depth,
    torch::Tensor volume_origin,
    torch::Tensor voxel_size,
    torch::Tensor img_size,
    torch::Tensor volume,
    torch::Tensor weight){
    const int volume_dim_x = volume.size(0);
    const int volume_dim_y = volume.size(1);
    const int volume_dim_z = volume.size(2);


    dim3 grid(NUM_BLOCKS(volume_dim_x), NUM_BLOCKS(volume_dim_y), NUM_BLOCKS(volume_dim_z));
    dim3 block(THREADS, THREADS, THREADS);

    integrate_kernel<<<grid, block>>>(
        intrinsic.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        pose.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        depth.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        volume_origin.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        voxel_size.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        img_size.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        volume.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        weight.packed_accessor32<float,3,torch::RestrictPtrTraits>());

    return ;
}
