/*
 * torch::Tensor wrappers for common.cu (moved out of the .cu so it stays torch-free; see common_cuda.h).
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "common.h"
#include "common_cuda.h"


at::Tensor sampleRaysUniformOccupiedVoxels(const at::Tensor z_in_out,  const at::Tensor z_sampled, at::Tensor z_vals)
{
  CHECK_INPUT(z_in_out);
  CHECK_INPUT(z_sampled);
  CHECK_INPUT(z_vals);
  AT_ASSERTM(z_vals.sizes()==z_sampled.sizes());

  const int N_rays = z_sampled.sizes()[0];
  const int N_samples = z_sampled.sizes()[1];

  AT_DISPATCH_FLOATING_TYPES(z_in_out.scalar_type(), "sample_rays_uniform_occupied_voxels_kernel", ([&]
  {
    launchSampleRaysUniformOccupiedVoxels<scalar_t>(z_sampled.data_ptr<scalar_t>(), z_in_out.data_ptr<scalar_t>(), z_vals.data_ptr<scalar_t>(), N_rays, N_samples, z_in_out.size(1), at::cuda::getCurrentCUDAStream());
  }));

  return z_vals;
}


at::Tensor postprocessOctreeRayTracing(const at::Tensor ray_index, const at::Tensor depth_in_out, const at::Tensor unique_intersect_ray_ids, const at::Tensor start_poss, const int max_intersections, const int N_rays)
{
  CHECK_INPUT(ray_index);
  CHECK_INPUT(depth_in_out);
  CHECK_INPUT(start_poss);
  const at::Tensor unique_ids = unique_intersect_ray_ids.contiguous();

  const int n_unique_ids = unique_ids.sizes()[0];
  at::Tensor depths_in_out_padded = at::zeros({N_rays,max_intersections,2}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(false));
  AT_DISPATCH_FLOATING_TYPES(depth_in_out.scalar_type(), "postprocessOctreeRayTracingKernel", ([&]
  {
    launchPostprocessOctreeRayTracing<scalar_t>(ray_index.data_ptr<int64_t>(), ray_index.size(0), depth_in_out.data_ptr<scalar_t>(), unique_ids.data_ptr<int64_t>(), start_poss.data_ptr<int64_t>(), n_unique_ids, depths_in_out_padded.data_ptr<scalar_t>(), max_intersections, at::cuda::getCurrentCUDAStream());
  }));

  return depths_in_out_padded;
}


at::Tensor calculateBarycentricCoordinate3D(const at::Tensor &triangle, const at::Tensor &p)
{
  auto vector_A = triangle[1]-triangle[2];
  auto vector_B = triangle[1]-triangle[0];
  auto normal = vector_A.cross(vector_B);
  auto areaABC = (normal * at::cross(triangle[1]-triangle[0], triangle[2]-triangle[0])).sum();
  auto areaPBC = (normal * at::cross(triangle[1]-p, triangle[2]-p)).sum();
  auto areaPCA = (normal * at::cross(triangle[2]-p, triangle[0]-p)).sum();
  at::Tensor ws = at::zeros({3}).to(torch::kFloat32);
  ws[0] = areaPBC / areaABC;
  ws[1] = areaPCA / areaABC;
  ws[2] = 1-ws[0]-ws[1];
  return ws;
}


void rayColorToTextureImageCUDA(const at::Tensor &F, const at::Tensor &V, const at::Tensor &hit_locations, const at::Tensor &hit_face_ids, const at::Tensor &uvs_tex, at::Tensor &uvs)
{
  CHECK_CONTIGUOUS(F);
  CHECK_CONTIGUOUS(V);
  CHECK_CONTIGUOUS(hit_locations);
  CHECK_CONTIGUOUS(hit_face_ids);
  CHECK_CONTIGUOUS(uvs_tex);
  CHECK_CONTIGUOUS(uvs);

  AT_DISPATCH_FLOATING_TYPES(V.scalar_type(), "rayColorToTextureImageKernel", ([&]
  {
    launchRayColorToTextureImage<scalar_t>(F.data_ptr<int64_t>(), V.data_ptr<scalar_t>(), hit_locations.data_ptr<scalar_t>(), hit_face_ids.data_ptr<int64_t>(), uvs_tex.data_ptr<scalar_t>(), uvs.data_ptr<scalar_t>(), int(hit_locations.size(0)), at::cuda::getCurrentCUDAStream());
  }));
}
