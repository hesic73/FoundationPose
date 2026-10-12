/*
 * Torch-free launchers for common.cu (kernels take raw pointers), so that common.cu does not include
 * PyTorch headers (nvcc's front end + a GCC 15 host compiler fails on ATen/core/List_inl.h).
 * The torch::Tensor wrappers live in common_torch.cpp.
 */
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// z_sampled/z_vals: [N_rays, N_samples], z_in_out: [N_rays, max_n_box, 2]
template<class scalar_t>
void launchSampleRaysUniformOccupiedVoxels(const scalar_t *z_sampled, const scalar_t *z_in_out, scalar_t *z_vals, int N_rays, int N_samples, int max_n_box, cudaStream_t stream);

// ray_index: [N_hits], depth_in_out: [N_hits, 2], unique_intersect_ray_ids/start_poss: [n_unique], out: [N_rays, max_intersections, 2]
template<class scalar_t>
void launchPostprocessOctreeRayTracing(const int64_t *ray_index, int N_hits, const scalar_t *depth_in_out, const int64_t *unique_intersect_ray_ids, const int64_t *start_poss, int n_unique, scalar_t *depths_in_out_padded, int max_intersections, cudaStream_t stream);

// F: [nF, 3] int64, V: [nV, 3], hit_locations: [N_hits, 3], hit_face_ids: [N_hits], uvs_tex: [nV, 2], uvs: [N_hits, 2]
template<class scalar_t>
void launchRayColorToTextureImage(const int64_t *F, const scalar_t *V, const scalar_t *hit_locations, const int64_t *hit_face_ids, const scalar_t *uvs_tex, scalar_t *uvs, int N_hits, cudaStream_t stream);
