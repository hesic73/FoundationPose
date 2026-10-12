/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */



#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <bits/stdc++.h>
#include <stdint.h>
#include <cstdio>
#include <stdint.h>
#include <stdexcept>
#include <limits>
#include <set>
#include "common_cuda.h"
#include "Eigen/Dense"

// Torch-free: the torch::Tensor wrappers are in common_torch.cpp (GCC 15 + nvcc cannot parse PyTorch headers).
static inline int divCeil(int a, int b)
{
  return (a+b-1)/b;
};





/**
 * @brief
 *
 * @tparam scalar_t
 * @param z_sampled
 * @param z_in_out
 * @param z_vals
 * @return __global__
 */
template <typename scalar_t>
__global__ void sample_rays_uniform_occupied_voxels_kernel(const scalar_t *z_sampled, const scalar_t *z_in_out, scalar_t *z_vals, const int N_rays, const int N_samples, const int max_n_box)
{
  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  const int i_sample = blockIdx.y * blockDim.y + threadIdx.y;
  if (i_ray>=N_rays) return;
  if (i_sample>=N_samples) return;

  int i_box = 0;
  float z_remain = z_sampled[i_ray*N_samples+i_sample];
  const scalar_t (*z_in_out_cur_ray)[2] = reinterpret_cast<const scalar_t (*)[2]>(z_in_out + i_ray*max_n_box*2);
  scalar_t *z_val = z_vals + i_ray*N_samples + i_sample;
  const float eps = 1e-4;

  if (z_in_out_cur_ray[0][0]==0) return;

  while (1)
  {
    if (i_box>=max_n_box)
    {
      if (z_remain<=eps)
      {
        *z_val = z_in_out_cur_ray[max_n_box-1][1];
      }
      else
      {
        printf("ERROR sample_rays_uniform_occupied_voxels_kernel: z_remain=%f, i_ray=%d, i_sample=%d, i_box=%d, z_in_out_cur_ray=(%f,%f)\n",z_remain,i_ray,i_sample,i_box,z_in_out_cur_ray[i_box][0],z_in_out_cur_ray[i_box][1]);
        for (int i=0;i<max_n_box;i++)
        {
          printf("z_in_out_cur_ray[%d]=(%f,%f)\n",i,z_in_out_cur_ray[i][0],z_in_out_cur_ray[i][1]);
        }
        while (1){};
      }


      return;
    }

    if (z_in_out_cur_ray[i_box][0]==0)
    {
      if (z_remain<=eps && i_box>=1)
      {
        *z_val = z_in_out_cur_ray[i_box-1][1];
        return;
      }
      else
      {
        printf("ERROR sample_rays_uniform_occupied_voxels_kernel: z_remain=%f, i_ray=%d, i_sample=%d, i_box=%d, z_in_out_cur_ray=(%f,%f)\n",z_remain,i_ray,i_sample,i_box,z_in_out_cur_ray[i_box][0],z_in_out_cur_ray[i_box][1]);
        for (int i=0;i<max_n_box;i++)
        {
          printf("z_in_out_cur_ray[%d]=(%f,%f)\n",i,z_in_out_cur_ray[i][0],z_in_out_cur_ray[i][1]);
        }
        while (1){};
      }
    }

    float box_len = z_in_out_cur_ray[i_box][1]-z_in_out_cur_ray[i_box][0];
    if (z_remain<=box_len)
    {
      *z_val = z_in_out_cur_ray[i_box][0] + z_remain;
      return;
    }
    z_remain -= box_len;
    i_box++;
  }
}

template<class scalar_t>
void launchSampleRaysUniformOccupiedVoxels(const scalar_t *z_sampled, const scalar_t *z_in_out, scalar_t *z_vals, int N_rays, int N_samples, int max_n_box, cudaStream_t stream)
{
  const int threadx = 32;
  const int thready = 32;
  dim3 blocks(divCeil(N_rays,threadx), divCeil(N_samples,thready));
  dim3 threads(threadx, thready);
  sample_rays_uniform_occupied_voxels_kernel<scalar_t><<<blocks, threads, 0, stream>>>(z_sampled, z_in_out, z_vals, N_rays, N_samples, max_n_box);
}
template void launchSampleRaysUniformOccupiedVoxels<float>(const float*, const float*, float*, int, int, int, cudaStream_t);
template void launchSampleRaysUniformOccupiedVoxels<double>(const double*, const double*, double*, int, int, int, cudaStream_t);

template<class scalar_t>
__global__ void postprocessOctreeRayTracingKernel(const int64_t *ray_index, const int N_hits, const scalar_t *depth_in_out_ptr, const int64_t *unique_intersect_ray_ids, const int64_t *start_poss, const int n_unique, scalar_t *depths_in_out_padded, const int max_intersections)
{
  const int unique_id_pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (unique_id_pos>=n_unique) return;
  const int i_ray = unique_intersect_ray_ids[unique_id_pos];

  int i_intersect = 0;
  const scalar_t (*depth_in_out)[2] = reinterpret_cast<const scalar_t (*)[2]>(depth_in_out_ptr);
  scalar_t (*cur_depths_in_out_padded)[2] = reinterpret_cast<scalar_t (*)[2]>(depths_in_out_padded + (int64_t)i_ray*max_intersections*2);
  for (int i=start_poss[unique_id_pos];i<N_hits;i++)
  {
    if (ray_index[i]!=i_ray) break;
    if (depth_in_out[i][0]==0 || depth_in_out[i][1]==0) break;
    if (depth_in_out[i][0]>depth_in_out[i][1]) continue;
    if (abs(depth_in_out[i][1]-depth_in_out[i][0])<1e-4) continue;

    cur_depths_in_out_padded[i_intersect][0] = depth_in_out[i][0];
    cur_depths_in_out_padded[i_intersect][1] = depth_in_out[i][1];

    i_intersect++;

  }
}

template<class scalar_t>
void launchPostprocessOctreeRayTracing(const int64_t *ray_index, int N_hits, const scalar_t *depth_in_out, const int64_t *unique_intersect_ray_ids, const int64_t *start_poss, int n_unique, scalar_t *depths_in_out_padded, int max_intersections, cudaStream_t stream)
{
  dim3 threads = {256};
  dim3 blocks = {(unsigned)divCeil(n_unique,threads.x)};
  postprocessOctreeRayTracingKernel<scalar_t><<<blocks,threads,0,stream>>>(ray_index, N_hits, depth_in_out, unique_intersect_ray_ids, start_poss, n_unique, depths_in_out_padded, max_intersections);
}
template void launchPostprocessOctreeRayTracing<float>(const int64_t*, int, const float*, const int64_t*, const int64_t*, int, float*, int, cudaStream_t);
template void launchPostprocessOctreeRayTracing<double>(const int64_t*, int, const double*, const int64_t*, const int64_t*, int, double*, int, cudaStream_t);


__device__ Eigen::Vector3f calculateBarycentricCoordinate3DKernel(const Eigen::Matrix<float,3,3> &triangle, const Eigen::Vector3f &p)
{
  Eigen::Vector3f vector_A = triangle.row(1)-triangle.row(2);
  Eigen::Vector3f vector_B = triangle.row(1)-triangle.row(0);
  Eigen::Vector3f normal = vector_A.cross(vector_B);
  float areaABC = (normal.array() * (triangle.row(1)-triangle.row(0)).cross(triangle.row(2)-triangle.row(0)).transpose().array()).sum();
  float areaPBC = (normal.array() * (triangle.row(1).transpose()-p).cross(triangle.row(2).transpose()-p).array()).sum();
  float areaPCA = (normal.array() * (triangle.row(2).transpose()-p).cross(triangle.row(0).transpose()-p).array()).sum();
  Eigen::Vector3f w;
  w[0] = areaPBC / areaABC;
  w[1] = areaPCA / areaABC;
  w[2] = 1-w[0]-w[1];
  return w;

}


__device__ void calculateBarycentricCoordinate2DKernel(const Eigen::Matrix<float,3,2> &triangle, const Eigen::Vector2f &p, Eigen::Vector3f &w)
{
  Eigen::Vector2f CA = triangle.row(0)-triangle.row(2);
  Eigen::Vector2f AC = -CA;
  Eigen::Vector2f CP = p-triangle.row(2).transpose();
  Eigen::Vector2f AB = triangle.row(1)-triangle.row(0);
  Eigen::Vector2f AP = p-triangle.row(0).transpose();
  Eigen::Matrix2f numerator, denominator;
  numerator << CA, CP;
  denominator<<AB, AC;
  w(1) = numerator.determinant()/denominator.determinant();
  numerator << AB, AP;
  w(2) = numerator.determinant()/denominator.determinant();
  w(0) = 1-w(1)-w(2);
}



template<class scalar_t>
__global__ void rayColorToTextureImageKernel(const int64_t *F_ptr, const scalar_t *V_ptr, const scalar_t *hit_locations_ptr, const int64_t *hit_face_ids, const scalar_t *uvs_tex_ptr, scalar_t *uvs_ptr, const int N_hits)
{
  const int i_hit = blockIdx.x*blockDim.x + threadIdx.x;
  if (i_hit>=N_hits) return;

  const int64_t (*F)[3] = reinterpret_cast<const int64_t (*)[3]>(F_ptr);
  const scalar_t (*V)[3] = reinterpret_cast<const scalar_t (*)[3]>(V_ptr);
  const scalar_t (*hit_locations)[3] = reinterpret_cast<const scalar_t (*)[3]>(hit_locations_ptr);
  const scalar_t (*uvs_tex)[2] = reinterpret_cast<const scalar_t (*)[2]>(uvs_tex_ptr);
  scalar_t (*uvs)[2] = reinterpret_cast<scalar_t (*)[2]>(uvs_ptr);
  auto face = F[hit_face_ids[i_hit]];
  Eigen::Matrix3f tri_v = Eigen::Matrix3f::Zero();
  for (int r=0;r<3;r++)
  {
    for (int c=0;c<3;c++)
    {
      tri_v(r,c) = V[face[r]][c];
    }
  }
  auto location = hit_locations[i_hit];
  Eigen::Vector3f p(location[0], location[1], location[2]);
  auto w = calculateBarycentricCoordinate3DKernel(tri_v, p);

  Eigen::Matrix<float,3,2> tri_uvs;
  for (int i=0;i<3;i++)
  {
    for (int j=0;j<2;j++)
    {
      tri_uvs(i,j) = uvs_tex[face[i]][j];
    }
  }

  Eigen::Vector2f cur_uv = tri_uvs.transpose() * w;
  uvs[i_hit][0] = cur_uv(0);
  uvs[i_hit][1] = cur_uv(1);
}



template<class scalar_t>
void launchRayColorToTextureImage(const int64_t *F, const scalar_t *V, const scalar_t *hit_locations, const int64_t *hit_face_ids, const scalar_t *uvs_tex, scalar_t *uvs, int N_hits, cudaStream_t stream)
{
  dim3 threads = {512};
  dim3 blocks = {(unsigned)divCeil(N_hits,threads.x)};
  rayColorToTextureImageKernel<scalar_t><<<blocks,threads,0,stream>>>(F, V, hit_locations, hit_face_ids, uvs_tex, uvs, N_hits);
}
template void launchRayColorToTextureImage<float>(const int64_t*, const float*, const float*, const int64_t*, const float*, float*, int, cudaStream_t);
template void launchRayColorToTextureImage<double>(const int64_t*, const double*, const double*, const int64_t*, const double*, double*, int, cudaStream_t);



__device__ bool isLineIntersectLine(Eigen::Vector2f l1p1, Eigen::Vector2f l1p2, Eigen::Vector2f l2p1, Eigen::Vector2f l2p2)
{
  float q = (l1p1(1) - l2p1(1)) * (l2p2(0) - l2p1(0)) - (l1p1(0) - l2p1(0)) * (l2p2(1) - l2p1(1));
  float d = (l1p2(0) - l1p1(0)) * (l2p2(1) - l2p1(1)) - (l1p2(1) - l1p1(1)) * (l2p2(0) - l2p1(0));

  if ( d == 0 )
  {
    return false;
  }

  float r = q / d;

  q = (l1p1(1) - l2p1(1)) * (l1p2(0) - l1p1(0)) - (l1p1(0) - l2p1(0)) * (l1p2(1) - l1p1(1));
  float s = q / d;

  if( r < 0 || r > 1 || s < 0 || s > 1 )
  {
    return false;
  }

  return true;
}

__device__ bool isLineIntersectSquare(const Eigen::Vector2f &la, const Eigen::Vector2f &lb, const Eigen::Vector2f &p0, const Eigen::Vector2f &p1, const Eigen::Vector2f &p2, const Eigen::Vector2f &p3)
{
  return isLineIntersectLine(la, lb, p0, p1) || isLineIntersectLine(la, lb, p1, p2) || isLineIntersectLine(la, lb, p2, p3) || isLineIntersectLine(la, lb, p3, p0);
}


__device__ bool isPixelInsideTriangle(const Eigen::Matrix<float,3,2> &triangle, Eigen::Vector2f p, Eigen::Vector3f &w)
{
  calculateBarycentricCoordinate2DKernel(triangle, p, w);
  for (int j=0;j<3;j++)
  {
    if (w(j)<0) return false;
  }
  return true;
}
