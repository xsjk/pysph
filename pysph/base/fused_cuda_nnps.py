"""Device-resident neighbor metadata contract for fused CUDA SPH stages."""

import os
from functools import cache
from dataclasses import dataclass

import numpy as np

_SCAN_KERNEL = None

CUDA_SOURCE = r"""
extern "C" {

	__global__ void fused_reset_int(int *values, int n)
	{
	    int i = blockIdx.x * blockDim.x + threadIdx.x;
	    if (i < n) {
	        values[i] = 0;
	    }
	}

	__global__ void fused_reset_uint(unsigned int *values, int n)
	{
	    int i = blockIdx.x * blockDim.x + threadIdx.x;
	    if (i < n) {
	        values[i] = 0u;
	    }
	}

	__global__ void fused_reset_hbucket_metadata(
	    int *cell_bucket_counts,
	    unsigned int *bucket_h_max_bits,
	    unsigned int *cell_bucket_h_max_bits,
	    int flat_total,
	    int bucket_count
	)
	{
	    int i = blockIdx.x * blockDim.x + threadIdx.x;
	    if (i < flat_total) {
	        cell_bucket_counts[i] = 0;
	        cell_bucket_h_max_bits[i] = 0u;
	    }
	    if (i < bucket_count) {
	        bucket_h_max_bits[i] = 0u;
	    }
	}

__global__ void fused_reset_convergence_flag(int *flag)
{
    flag[0] = 1;
}

	__global__ void fused_reduce_max_float(const float *values, int n, float *out)
	{
    __shared__ float scratch[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float value = -3.402823466e+38F;
    if (i < n) {
        value = values[i];
    }
    scratch[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[blockIdx.x] = scratch[0];
	    }
	}

	__global__ void fused_reduce_min_float(const float *values, int n, float *out)
	{
	    __shared__ float scratch[256];
	    int tid = threadIdx.x;
	    int i = blockIdx.x * blockDim.x + tid;
	    float value = 3.402823466e+38F;
	    if (i < n) {
	        value = values[i];
	    }
	    scratch[tid] = value;
	    __syncthreads();
	    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
	        if (tid < stride) {
	            scratch[tid] = fminf(scratch[tid], scratch[tid + stride]);
	        }
	        __syncthreads();
	    }
	    if (tid == 0) {
	        out[blockIdx.x] = scratch[0];
	    }
	}

__global__ void fused_wrap_periodic_xyz(
    float *x,
    float *y,
    float *z,
    int n,
    float xmin,
    float xmax,
    float ymin,
    float ymax,
    float zmin,
    float zmax,
    int periodic_x,
    int periodic_y,
    int periodic_z
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    if (periodic_x) {
        float length = xmax - xmin;
        if (x[i] < xmin) {
            x[i] += length;
        }
        if (x[i] > xmax) {
            x[i] -= length;
        }
    }
    if (periodic_y) {
        float length = ymax - ymin;
        if (y[i] < ymin) {
            y[i] += length;
        }
        if (y[i] > ymax) {
            y[i] -= length;
        }
    }
    if (periodic_z) {
        float length = zmax - zmin;
        if (z[i] < zmin) {
            z[i] += length;
        }
        if (z[i] > zmax) {
            z[i] -= length;
        }
    }
}

__device__ int fused_clamp_cell(float value, float lower, float upper, int count)
{
    float span = upper - lower;
    float rel = (value - lower) / span;
    int cell = (int)floorf(rel * (float)count);
    if (cell < 0) {
        cell = 0;
    }
    if (cell >= count) {
        cell = count - 1;
    }
    return cell;
}

__device__ int fused_linear_cell(int cx, int cy, int cz, int nx, int ny)
{
    return ((cz * ny) + cy) * nx + cx;
}

__device__ bool fused_valid_offset(int offset, int count)
{
    if (count == 1) {
        return offset == 0;
    }
    if (count == 2) {
        return offset <= 0;
    }
    return true;
}

__device__ int fused_wrapped_cell(int cell, int count)
{
    if (cell < 0) {
        return cell + count;
    }
    if (cell >= count) {
        return cell - count;
    }
    return cell;
}

__device__ bool fused_neighbor_cell(
    int base,
    int offset,
    int count,
    int periodic,
    int *out
)
{
    int cell = base + offset;
    if (periodic) {
        *out = fused_wrapped_cell(cell, count);
        return true;
    }
    if (cell < 0 || cell >= count) {
        return false;
    }
    *out = cell;
    return true;
}

__device__ float fused_minimum_image(float delta, float length, int periodic)
{
    if (periodic) {
        float half = 0.5f * length;
        if (delta > half) {
            delta -= length;
        }
        if (delta < -half) {
            delta += length;
        }
    }
    return delta;
}

	__device__ bool fused_in_support_xyz(
    int dst,
    int src,
    const float *x,
    const float *y,
    const float *z,
    const float *h,
    float xmin,
    float xmax,
    float ymin,
    float ymax,
    float zmin,
    float zmax,
    int periodic_x,
    int periodic_y,
    int periodic_z,
    float radius_scale
)
{
    float dx = x[src] - x[dst];
    float dy = y[src] - y[dst];
    float dz = z[src] - z[dst];
    dx = fused_minimum_image(dx, xmax - xmin, periodic_x);
    dy = fused_minimum_image(dy, ymax - ymin, periodic_y);
    dz = fused_minimum_image(dz, zmax - zmin, periodic_z);
    float dist2 = dx * dx + dy * dy + dz * dz;
	    float support = radius_scale * fmaxf(h[dst], h[src]);
	    return dist2 < support * support;
	}

	__device__ float fused_axis_cell_distance(
	    float point,
	    int cell,
	    float lower,
	    float upper,
	    float width,
	    int periodic
	)
	{
	    float center = lower + ((float)cell + 0.5f) * width;
	    float delta = fused_minimum_image(center - point, upper - lower, periodic);
	    float distance = fabsf(delta) - 0.5f * width;
	    if (distance < 0.0f) {
	        distance = 0.0f;
	    }
	    return distance;
	}

	__device__ float fused_cell_distance2_to_particle(
	    int cell,
	    int nx,
	    int ny,
	    float px,
	    float py,
	    float pz,
	    float xmin,
	    float xmax,
	    float ymin,
	    float ymax,
	    float zmin,
	    float zmax,
	    int periodic_x,
	    int periodic_y,
	    int periodic_z,
	    float cell_width_x,
	    float cell_width_y,
	    float cell_width_z
	)
	{
	    int cx = cell % nx;
	    int tmp = cell / nx;
	    int cy = tmp % ny;
	    int cz = tmp / ny;
	    float dx = fused_axis_cell_distance(
	        px, cx, xmin, xmax, cell_width_x, periodic_x
	    );
	    float dy = fused_axis_cell_distance(
	        py, cy, ymin, ymax, cell_width_y, periodic_y
	    );
	    float dz = fused_axis_cell_distance(
	        pz, cz, zmin, zmax, cell_width_z, periodic_z
	    );
	    return dx * dx + dy * dy + dz * dz;
	}

__global__ void fused_compute_cell_ids_counts_xyz(
    const float *x,
    const float *y,
    const float *z,
    int n,
    const float *lower,
    const float *upper,
    int nx,
    int ny,
    int nz,
    int *cell_counts,
    int *particle_cell,
    int *particle_local_index
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int cx = fused_clamp_cell(x[i], lower[0], upper[0], nx);
        int cy = fused_clamp_cell(y[i], lower[1], upper[1], ny);
        int cz = fused_clamp_cell(z[i], lower[2], upper[2], nz);
        int cell = fused_linear_cell(cx, cy, cz, nx, ny);
        particle_cell[i] = cell;
        particle_local_index[i] = atomicAdd(&cell_counts[cell], 1);
    }
}

	__global__ void fused_scatter_sorted_particles(
    int n,
    const int *particle_cell,
    const int *particle_local_index,
    const int *cell_starts,
    int *sorted_ids
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int cell = particle_cell[i];
        int out = cell_starts[cell] + particle_local_index[i];
        sorted_ids[out] = i;
	    }
	}

	__device__ int fused_hbucket_index(float hi, float h_min, int bucket_count)
	{
	    float ratio = fmaxf(hi / h_min, 1.0f);
	    int bucket = (int)floorf(log2f(ratio));
	    if (bucket < 0) {
	        bucket = 0;
	    }
	    if (bucket >= bucket_count) {
	        bucket = bucket_count - 1;
	    }
	    return bucket;
	}

	__global__ void fused_compute_hbucket_ids_counts_xyz(
	    const float *x,
	    const float *y,
	    const float *z,
	    const float *h,
	    int n,
	    const float *lower,
	    const float *upper,
	    float h_min,
	    int bucket_count,
	    int nx,
	    int ny,
	    int nz,
	    int total_cells,
	    int *cell_bucket_counts,
	    int *particle_cell,
	    int *particle_bucket,
	    int *particle_local_index,
	    unsigned int *bucket_h_max_bits,
	    unsigned int *cell_bucket_h_max_bits
	)
	{
	    int i = blockIdx.x * blockDim.x + threadIdx.x;
	    if (i < n) {
	        int cx = fused_clamp_cell(x[i], lower[0], upper[0], nx);
	        int cy = fused_clamp_cell(y[i], lower[1], upper[1], ny);
	        int cz = fused_clamp_cell(z[i], lower[2], upper[2], nz);
	        int cell = fused_linear_cell(cx, cy, cz, nx, ny);
	        int bucket = fused_hbucket_index(h[i], h_min, bucket_count);
	        int flat = bucket * total_cells + cell;
	        particle_cell[i] = cell;
	        particle_bucket[i] = bucket;
	        particle_local_index[i] = atomicAdd(&cell_bucket_counts[flat], 1);
	        atomicMax(&bucket_h_max_bits[bucket], __float_as_uint(h[i]));
	        atomicMax(&cell_bucket_h_max_bits[flat], __float_as_uint(h[i]));
	    }
	}

	__global__ void fused_scatter_hbucket_sorted_particles(
	    int n,
	    int total_cells,
	    const int *particle_cell,
	    const int *particle_bucket,
	    const int *particle_local_index,
	    const int *cell_bucket_starts,
	    int *sorted_ids
	)
	{
	    int i = blockIdx.x * blockDim.x + threadIdx.x;
	    if (i < n) {
	        int flat = particle_bucket[i] * total_cells + particle_cell[i];
	        int out = cell_bucket_starts[flat] + particle_local_index[i];
	        sorted_ids[out] = i;
	    }
	}

__global__ void fused_cluster_counts_from_cells(
    int total_cells,
    const int *cell_counts,
    int cluster_size,
    int *cell_cluster_count
)
{
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell < total_cells) {
        int count = cell_counts[cell];
        cell_cluster_count[cell] = (count + cluster_size - 1) / cluster_size;
    }
}

__global__ void fused_fill_destination_clusters(
    int total_cells,
    int cluster_size,
    const int *cell_counts,
    const int *cell_starts,
    const int *cell_cluster_start,
    int *cluster_cell,
    int *cluster_begin,
    int *cluster_count
)
{
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= total_cells) {
        return;
    }
    int count = cell_counts[cell];
    int clusters = (count + cluster_size - 1) / cluster_size;
    int first_cluster = cell_cluster_start[cell];
    for (int local = 0; local < clusters; ++local) {
        int cluster = first_cluster + local;
        int live = count - local * cluster_size;
        if (live > cluster_size) {
            live = cluster_size;
        }
        cluster_cell[cluster] = cell;
        cluster_begin[cluster] = cell_starts[cell] + local * cluster_size;
        cluster_count[cluster] = live;
    }
}

__global__ void fused_last_cluster_total(
    int total_cells,
    const int *cell_cluster_start,
    const int *cell_cluster_count,
    int *cluster_total
)
{
    int last = total_cells - 1;
    cluster_total[0] = cell_cluster_start[last] + cell_cluster_count[last];
}

	__global__ void fused_count_neighbors_from_context(
    const float *x,
    const float *y,
    const float *z,
    const float *h,
    int n,
    float xmin,
    float xmax,
    float ymin,
    float ymax,
    float zmin,
    float zmax,
    int periodic_x,
    int periodic_y,
    int periodic_z,
    float radius_scale,
    int search_radius_cells,
    int nx,
    int ny,
    int nz,
    const int *cell_counts,
    const int *cell_starts,
    const int *sorted_ids,
    int *neighbor_counts
)
{
    int dst = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst >= n) {
        return;
    }
    int base_cx = fused_clamp_cell(x[dst], xmin, xmax, nx);
    int base_cy = fused_clamp_cell(y[dst], ymin, ymax, ny);
    int base_cz = fused_clamp_cell(z[dst], zmin, zmax, nz);
    int count = 0;
    for (int oz = -search_radius_cells; oz <= search_radius_cells; ++oz) {
        if (!fused_valid_offset(oz, nz)) {
            continue;
        }
        int cz = 0;
        if (!fused_neighbor_cell(base_cz, oz, nz, periodic_z, &cz)) {
            continue;
        }
        for (int oy = -search_radius_cells; oy <= search_radius_cells; ++oy) {
            if (!fused_valid_offset(oy, ny)) {
                continue;
            }
            int cy = 0;
            if (!fused_neighbor_cell(base_cy, oy, ny, periodic_y, &cy)) {
                continue;
            }
            for (int ox = -search_radius_cells; ox <= search_radius_cells; ++ox) {
                if (!fused_valid_offset(ox, nx)) {
                    continue;
                }
                int cx = 0;
                if (!fused_neighbor_cell(base_cx, ox, nx, periodic_x, &cx)) {
                    continue;
                }
                int cell = fused_linear_cell(cx, cy, cz, nx, ny);
                int begin = cell_starts[cell];
                int end = begin + cell_counts[cell];
                for (int pos = begin; pos < end; ++pos) {
                    int src = sorted_ids[pos];
                    if (fused_in_support_xyz(
                        dst,
                        src,
                        x,
                        y,
                        z,
                        h,
                        xmin,
                        xmax,
                        ymin,
                        ymax,
                        zmin,
                        zmax,
                        periodic_x,
                        periodic_y,
                        periodic_z,
                        radius_scale
                    )) {
                        count += 1;
                    }
                }
            }
        }
    }
	    neighbor_counts[dst] = count;
	}

	__global__ void fused_count_neighbors_from_hbucket_context(
	    const float *x,
	    const float *y,
	    const float *z,
	    const float *h,
	    int n,
	    float xmin,
	    float xmax,
	    float ymin,
	    float ymax,
	    float zmin,
	    float zmax,
	    int periodic_x,
	    int periodic_y,
	    int periodic_z,
	    float radius_scale,
	    int nx,
	    int ny,
	    int nz,
	    int total_cells,
	    int bucket_count,
	    float cell_width_x,
	    float cell_width_y,
	    float cell_width_z,
	    const unsigned int *bucket_h_max_bits,
	    const int *cell_bucket_counts,
	    const int *cell_bucket_starts,
	    const int *sorted_ids,
	    int *neighbor_counts
	)
	{
	    int dst = blockIdx.x * blockDim.x + threadIdx.x;
	    if (dst >= n) {
	        return;
	    }
	    int base_cx = fused_clamp_cell(x[dst], xmin, xmax, nx);
	    int base_cy = fused_clamp_cell(y[dst], ymin, ymax, ny);
	    int base_cz = fused_clamp_cell(z[dst], zmin, zmax, nz);
	    float dst_h = h[dst];
	    int count = 0;
	    for (int bucket = 0; bucket < bucket_count; ++bucket) {
	        float bucket_h = __uint_as_float(bucket_h_max_bits[bucket]);
	        if (bucket_h <= 0.0f) {
	            continue;
	        }
	        float support = radius_scale * fmaxf(dst_h, bucket_h);
	        float support2 = support * support;
	        int max_x = (int)ceilf(support / cell_width_x);
	        int max_y = (int)ceilf(support / cell_width_y);
	        int max_z = (int)ceilf(support / cell_width_z);
	        int full_x = periodic_x && nx <= 2 * max_x + 1;
	        int full_y = periodic_y && ny <= 2 * max_y + 1;
	        int full_z = periodic_z && nz <= 2 * max_z + 1;
	        int loops_x = full_x ? nx : 2 * max_x + 1;
	        int loops_y = full_y ? ny : 2 * max_y + 1;
	        int loops_z = full_z ? nz : 2 * max_z + 1;
	        for (int iz = 0; iz < loops_z; ++iz) {
	            int cz = iz;
	            if (!full_z) {
	                int offset = iz - max_z;
	                if (!fused_neighbor_cell(base_cz, offset, nz, periodic_z, &cz)) {
	                    continue;
	                }
	            }
	            for (int iy = 0; iy < loops_y; ++iy) {
	                int cy = iy;
	                if (!full_y) {
	                    int offset = iy - max_y;
	                    if (!fused_neighbor_cell(base_cy, offset, ny, periodic_y, &cy)) {
	                        continue;
	                    }
	                }
	                for (int ix = 0; ix < loops_x; ++ix) {
	                    int cx = ix;
	                    if (!full_x) {
	                        int offset = ix - max_x;
	                        if (!fused_neighbor_cell(base_cx, offset, nx, periodic_x, &cx)) {
	                            continue;
	                        }
	                    }
	                    int cell = fused_linear_cell(cx, cy, cz, nx, ny);
	                    float cell_distance2 = fused_cell_distance2_to_particle(
	                        cell,
	                        nx,
	                        ny,
	                        x[dst],
	                        y[dst],
	                        z[dst],
	                        xmin,
	                        xmax,
	                        ymin,
	                        ymax,
	                        zmin,
	                        zmax,
	                        periodic_x,
	                        periodic_y,
	                        periodic_z,
	                        cell_width_x,
	                        cell_width_y,
	                        cell_width_z
	                    );
	                    if (cell_distance2 > support2) {
	                        continue;
	                    }
	                    int flat = bucket * total_cells + cell;
	                    int begin = cell_bucket_starts[flat];
	                    int end = begin + cell_bucket_counts[flat];
	                    for (int pos = begin; pos < end; ++pos) {
	                        int src = sorted_ids[pos];
	                        if (fused_in_support_xyz(
	                            dst,
	                            src,
	                            x,
	                            y,
	                            z,
	                            h,
	                            xmin,
	                            xmax,
	                            ymin,
	                            ymax,
	                            zmin,
	                            zmax,
	                            periodic_x,
	                            periodic_y,
	                            periodic_z,
	                            radius_scale
	                        )) {
	                            count += 1;
	                        }
	                    }
	                }
	            }
	        }
	    }
	    neighbor_counts[dst] = count;
	}

	__global__ void fused_count_hbucket_traversal_work(
	    const float *x,
	    const float *y,
	    const float *z,
	    const float *h,
	    int n,
	    float xmin,
	    float xmax,
	    float ymin,
	    float ymax,
	    float zmin,
	    float zmax,
	    int periodic_x,
	    int periodic_y,
	    int periodic_z,
	    float radius_scale,
	    int nx,
	    int ny,
	    int nz,
	    int total_cells,
	    int bucket_count,
	    float cell_width_x,
	    float cell_width_y,
	    float cell_width_z,
	    const unsigned int *bucket_h_max_bits,
	    const unsigned int *cell_bucket_h_max_bits,
	    const int *cell_bucket_counts,
	    const int *cell_bucket_starts,
	    const int *sorted_ids,
	    int *visited_cell_counts,
	    int *candidate_counts,
	    int *neighbor_counts
	)
	{
	    int dst = blockIdx.x * blockDim.x + threadIdx.x;
	    if (dst >= n) {
	        return;
	    }
	    float dst_x = x[dst];
	    float dst_y = y[dst];
	    float dst_z = z[dst];
	    float dst_h = h[dst];
	    int base_cx = fused_clamp_cell(dst_x, xmin, xmax, nx);
	    int base_cy = fused_clamp_cell(dst_y, ymin, ymax, ny);
	    int base_cz = fused_clamp_cell(dst_z, zmin, zmax, nz);
	    int visited_cells = 0;
	    int candidates = 0;
	    int neighbors = 0;
	    for (int bucket = 0; bucket < bucket_count; ++bucket) {
	        float bucket_h = __uint_as_float(bucket_h_max_bits[bucket]);
	        if (bucket_h <= 0.0f) {
	            continue;
	        }
	        float bucket_support = radius_scale * fmaxf(dst_h, bucket_h);
	        int max_x = (int)ceilf(bucket_support / cell_width_x);
	        int max_y = (int)ceilf(bucket_support / cell_width_y);
	        int max_z = (int)ceilf(bucket_support / cell_width_z);
	        int full_x = periodic_x && nx <= 2 * max_x + 1;
	        int full_y = periodic_y && ny <= 2 * max_y + 1;
	        int full_z = periodic_z && nz <= 2 * max_z + 1;
	        int loops_x = full_x ? nx : 2 * max_x + 1;
	        int loops_y = full_y ? ny : 2 * max_y + 1;
	        int loops_z = full_z ? nz : 2 * max_z + 1;
	        for (int iz = 0; iz < loops_z; ++iz) {
	            int cz = iz;
	            if (!full_z) {
	                int offset = iz - max_z;
	                if (!fused_neighbor_cell(base_cz, offset, nz, periodic_z, &cz)) {
	                    continue;
	                }
	            }
	            for (int iy = 0; iy < loops_y; ++iy) {
	                int cy = iy;
	                if (!full_y) {
	                    int offset = iy - max_y;
	                    if (!fused_neighbor_cell(base_cy, offset, ny, periodic_y, &cy)) {
	                        continue;
	                    }
	                }
	                for (int ix = 0; ix < loops_x; ++ix) {
	                    int cx = ix;
	                    if (!full_x) {
	                        int offset = ix - max_x;
	                        if (!fused_neighbor_cell(base_cx, offset, nx, periodic_x, &cx)) {
	                            continue;
	                        }
	                    }
	                    int cell = fused_linear_cell(cx, cy, cz, nx, ny);
	                    int flat = bucket * total_cells + cell;
	                    float cell_bucket_h = __uint_as_float(cell_bucket_h_max_bits[flat]);
	                    if (cell_bucket_h <= 0.0f) {
	                        continue;
	                    }
	                    float cell_support = radius_scale * fmaxf(dst_h, cell_bucket_h);
	                    float cell_support2 = cell_support * cell_support;
	                    float cell_distance2 = fused_cell_distance2_to_particle(
	                        cell,
	                        nx,
	                        ny,
	                        dst_x,
	                        dst_y,
	                        dst_z,
	                        xmin,
	                        xmax,
	                        ymin,
	                        ymax,
	                        zmin,
	                        zmax,
	                        periodic_x,
	                        periodic_y,
	                        periodic_z,
	                        cell_width_x,
	                        cell_width_y,
	                        cell_width_z
	                    );
	                    if (cell_distance2 > cell_support2) {
	                        continue;
	                    }
	                    int begin = cell_bucket_starts[flat];
	                    int end = begin + cell_bucket_counts[flat];
	                    visited_cells += 1;
	                    candidates += end - begin;
	                    for (int pos = begin; pos < end; ++pos) {
	                        int src = sorted_ids[pos];
	                        if (fused_in_support_xyz(
	                            dst,
	                            src,
	                            x,
	                            y,
	                            z,
	                            h,
	                            xmin,
	                            xmax,
	                            ymin,
	                            ymax,
	                            zmin,
	                            zmax,
	                            periodic_x,
	                            periodic_y,
	                            periodic_z,
	                            radius_scale
	                        )) {
	                            neighbors += 1;
	                        }
	                    }
	                }
	            }
	        }
	    }
	    visited_cell_counts[dst] = visited_cells;
	    candidate_counts[dst] = candidates;
	    neighbor_counts[dst] = neighbors;
	}

}
"""


@dataclass(frozen=True)
class FusedCudaNeighborContext:
    """Opaque neighbor metadata shared by fused CUDA stages.

    The context stores separate coordinate device arrays because that is the
    layout used by PySPH particle arrays and by the fastest prototype path.
    """

    n: int
    x: object
    y: object
    z: object
    h: object
    lower: np.ndarray
    upper: np.ndarray
    periodic: np.ndarray
    radius_scale: np.float32
    search_radius_cells: np.int32
    cell_counts: np.ndarray
    total_cells: int
    cluster_total: int
    stream: object
    sorted_ids: object
    cell_starts: object
    cell_particle_counts: object
    cluster_cell: object
    cluster_begin: object
    cluster_count: object
    timings_ms: tuple[tuple[str, float], ...]

    def __post_init__(self):
        """Validate the metadata contract used by fused kernels."""
        assert self.n > 0
        assert self.lower.dtype == np.float32
        assert self.upper.dtype == np.float32
        assert self.periodic.dtype == np.bool_
        assert isinstance(self.search_radius_cells, np.int32)
        assert self.search_radius_cells >= np.int32(1)
        assert self.cell_counts.dtype == np.int32
        assert self.total_cells == int(np.prod(self.cell_counts))
        assert self.cluster_total > 0
        assert self.lower.shape == (3,)
        assert self.upper.shape == (3,)
        assert self.periodic.shape == (3,)
        assert self.cell_counts.shape == (3,)
        assert self.x.dtype == np.float32
        assert self.y.dtype == np.float32
        assert self.z.dtype == np.float32
        assert self.h.dtype == np.float32
        assert self.sorted_ids.dtype == np.int32
        assert self.cell_starts.dtype == np.int32
        assert self.cell_particle_counts.dtype == np.int32
        assert self.cluster_cell.dtype == np.int32
        assert self.cluster_begin.dtype == np.int32
        assert self.cluster_count.dtype == np.int32

    @property
    def materializes_csr(self):
        """Return whether this context owns per-particle CSR neighbor lists."""
        return False

    @property
    def coordinate_layout(self):
        """Return the coordinate array ABI used by fused pair kernels."""
        return "separate_xyz"

    @property
    def uses_device_metadata(self):
        """Return whether traversal metadata is represented by device arrays."""
        return all(
            hasattr(array, "gpudata")
            for array in (
                self.sorted_ids,
                self.cell_starts,
                self.cell_particle_counts,
                self.cluster_cell,
                self.cluster_begin,
                self.cluster_count,
            )
        )

    @property
    def cell_count_tuple(self):
        """Return integer cell counts in x/y/z order."""
        return tuple(int(value) for value in self.cell_counts)

    @property
    def timing_total_ms(self):
        """Return the total measured metadata build time."""
        return sum(value for _name, value in self.timings_ms)


@dataclass(frozen=True)
class FusedCudaHBucketNeighborContext:
    """Device h-bucket metadata shared by fused CUDA pair stages."""

    n: int
    x: object
    y: object
    z: object
    h: object
    lower: np.ndarray
    upper: np.ndarray
    periodic: np.ndarray
    radius_scale: np.float32
    cell_counts: np.ndarray
    total_cells: int
    bucket_count: int
    cell_width: np.ndarray
    stream: object
    bucket_h_max_bits: object
    cell_bucket_h_max_bits: object
    sorted_ids: object
    cell_bucket_starts: object
    cell_bucket_counts: object
    timings_ms: tuple[tuple[str, float], ...]

    def __post_init__(self):
        """Validate the h-bucket metadata contract used by fused kernels."""
        assert self.n > 0
        assert self.lower.dtype == np.float32
        assert self.upper.dtype == np.float32
        assert self.periodic.dtype == np.bool_
        assert isinstance(self.radius_scale, np.float32)
        assert self.cell_counts.dtype == np.int32
        assert self.cell_width.dtype == np.float32
        assert self.total_cells == int(np.prod(self.cell_counts))
        assert self.bucket_count > 0
        assert self.lower.shape == (3,)
        assert self.upper.shape == (3,)
        assert self.periodic.shape == (3,)
        assert self.cell_counts.shape == (3,)
        assert self.cell_width.shape == (3,)
        assert self.x.dtype == np.float32
        assert self.y.dtype == np.float32
        assert self.z.dtype == np.float32
        assert self.h.dtype == np.float32
        assert self.bucket_h_max_bits.dtype == np.uint32
        assert self.cell_bucket_h_max_bits.dtype == np.uint32
        assert self.sorted_ids.dtype == np.int32
        assert self.cell_bucket_starts.dtype == np.int32
        assert self.cell_bucket_counts.dtype == np.int32

    @property
    def materializes_csr(self):
        """Return whether this context owns per-particle CSR neighbor lists."""
        return False

    @property
    def coordinate_layout(self):
        """Return the coordinate array ABI used by fused pair kernels."""
        return "separate_xyz"

    @property
    def uses_device_metadata(self):
        """Return whether traversal metadata is represented by device arrays."""
        return all(
            hasattr(array, "gpudata")
            for array in (
                self.bucket_h_max_bits,
                self.cell_bucket_h_max_bits,
                self.sorted_ids,
                self.cell_bucket_starts,
                self.cell_bucket_counts,
            )
        )

    @property
    def cell_count_tuple(self):
        """Return integer cell counts in x/y/z order."""
        return tuple(int(value) for value in self.cell_counts)

    @property
    def timing_total_ms(self):
        """Return the total measured metadata build time."""
        return sum(value for _name, value in self.timings_ms)


class FusedCudaNeighborWorkspace:
    """Reusable device buffers for sorted-cell metadata builds."""

    def __init__(self):
        self.lower = None
        self.upper = None
        self.cell_particle_counts = None
        self.cell_starts = None
        self.sorted_ids = None
        self.particle_cell = None
        self.particle_local_index = None
        self.cluster_cell = None
        self.cluster_begin = None
        self.cluster_count = None
        self.hbucket_cell_bucket_counts = None
        self.hbucket_cell_bucket_starts = None
        self.hbucket_sorted_ids = None
        self.hbucket_particle_cell = None
        self.hbucket_particle_bucket = None
        self.hbucket_particle_local_index = None
        self.hbucket_bucket_h_max_bits = None
        self.hbucket_cell_bucket_h_max_bits = None
        self.hbucket_h_min_ref = None

    def ensure_base(self, n: int, total_cells: int) -> None:
        import pycuda.gpuarray as gpuarray

        self.lower = _ensure_gpu_array(self.lower, 3, np.float32)
        self.upper = _ensure_gpu_array(self.upper, 3, np.float32)
        self.cell_particle_counts = _ensure_gpu_array(
            self.cell_particle_counts, total_cells, np.int32
        )
        self.cell_starts = _ensure_gpu_array(self.cell_starts, total_cells, np.int32)
        self.sorted_ids = _ensure_gpu_array(self.sorted_ids, n, np.int32)
        self.particle_cell = _ensure_gpu_array(self.particle_cell, n, np.int32)
        self.particle_local_index = _ensure_gpu_array(
            self.particle_local_index, n, np.int32
        )
        if self.cluster_cell is None:
            self.cluster_cell = gpuarray.empty((1,), np.int32)
            self.cluster_begin = gpuarray.empty((1,), np.int32)
            self.cluster_count = gpuarray.empty((1,), np.int32)

    def ensure_hbucket(self, n: int, flat_total: int, bucket_count: int) -> None:
        self.lower = _ensure_gpu_array(self.lower, 3, np.float32)
        self.upper = _ensure_gpu_array(self.upper, 3, np.float32)
        self.hbucket_cell_bucket_counts = _ensure_gpu_array(
            self.hbucket_cell_bucket_counts, flat_total, np.int32
        )
        self.hbucket_cell_bucket_starts = _ensure_gpu_array(
            self.hbucket_cell_bucket_starts, flat_total, np.int32
        )
        self.hbucket_sorted_ids = _ensure_gpu_array(
            self.hbucket_sorted_ids, n, np.int32
        )
        self.hbucket_particle_cell = _ensure_gpu_array(
            self.hbucket_particle_cell, n, np.int32
        )
        self.hbucket_particle_bucket = _ensure_gpu_array(
            self.hbucket_particle_bucket, n, np.int32
        )
        self.hbucket_particle_local_index = _ensure_gpu_array(
            self.hbucket_particle_local_index, n, np.int32
        )
        self.hbucket_bucket_h_max_bits = _ensure_gpu_array(
            self.hbucket_bucket_h_max_bits, bucket_count, np.uint32
        )
        self.hbucket_cell_bucket_h_max_bits = _ensure_gpu_array(
            self.hbucket_cell_bucket_h_max_bits, flat_total, np.uint32
        )


def build_fused_cuda_context_from_device_arrays(
    x: object,
    y: object,
    z: object,
    h: object,
    n: int,
    lower: np.ndarray,
    upper: np.ndarray,
    periodic: np.ndarray,
    radius_scale: np.float32,
    cell_counts: np.ndarray,
    stream: object,
    cluster_size: int,
) -> FusedCudaNeighborContext:
    """Build a no-CSR sorted-cell context from PySPH-style device arrays."""
    workspace = FusedCudaNeighborWorkspace()
    return build_fused_cuda_context_with_workspace(
        x,
        y,
        z,
        h,
        n,
        lower,
        upper,
        periodic,
        radius_scale,
        cell_counts,
        stream,
        cluster_size,
        workspace,
        True,
    )


def build_fused_cuda_context_with_workspace(
    x: object,
    y: object,
    z: object,
    h: object,
    n: int,
    lower: np.ndarray,
    upper: np.ndarray,
    periodic: np.ndarray,
    radius_scale: np.float32,
    cell_counts: np.ndarray,
    stream: object,
    cluster_size: int,
    workspace: FusedCudaNeighborWorkspace,
    build_cluster_metadata: bool,
) -> FusedCudaNeighborContext:
    """Build sorted-cell context using caller-owned reusable device buffers."""
    _ensure_cuda_context()
    assert x.dtype == np.float32
    assert y.dtype == np.float32
    assert z.dtype == np.float32
    assert h.dtype == np.float32
    assert n > 0
    assert lower.dtype == np.float32
    assert upper.dtype == np.float32
    assert periodic.dtype == np.bool_
    assert isinstance(radius_scale, np.float32)
    assert cell_counts.dtype == np.int32
    assert cluster_size > 0

    import pycuda.driver as cuda

    total_cells = int(np.prod(cell_counts))
    workspace.ensure_base(n, total_cells)
    d_lower = workspace.lower
    d_upper = workspace.upper
    d_cell_particle_counts = workspace.cell_particle_counts
    d_cell_starts = workspace.cell_starts
    d_sorted_ids = workspace.sorted_ids
    d_particle_cell = workspace.particle_cell
    d_particle_local_index = workspace.particle_local_index
    cuda.memcpy_htod_async(_device_ptr(d_lower), np.ascontiguousarray(lower), stream)
    cuda.memcpy_htod_async(_device_ptr(d_upper), np.ascontiguousarray(upper), stream)

    kernels = _module()
    reset_int = kernels.get_function("fused_reset_int")
    compute_cell_ids_counts = kernels.get_function("fused_compute_cell_ids_counts_xyz")
    scatter_sorted_particles = kernels.get_function("fused_scatter_sorted_particles")

    start, stop = _event_pair(stream)
    reset_int(
        _device_ptr(d_cell_particle_counts),
        np.int32(total_cells),
        block=(256, 1, 1),
        grid=_grid_size(total_cells),
        stream=stream,
    )
    compute_cell_ids_counts(
        _device_ptr(x),
        _device_ptr(y),
        _device_ptr(z),
        np.int32(n),
        _device_ptr(d_lower),
        _device_ptr(d_upper),
        np.int32(cell_counts[0]),
        np.int32(cell_counts[1]),
        np.int32(cell_counts[2]),
        _device_ptr(d_cell_particle_counts),
        _device_ptr(d_particle_cell),
        _device_ptr(d_particle_local_index),
        block=(256, 1, 1),
        grid=_grid_size(n),
        stream=stream,
    )
    _scan_int32(d_cell_particle_counts, d_cell_starts, stream)
    scatter_sorted_particles(
        np.int32(n),
        _device_ptr(d_particle_cell),
        _device_ptr(d_particle_local_index),
        _device_ptr(d_cell_starts),
        _device_ptr(d_sorted_ids),
        block=(256, 1, 1),
        grid=_grid_size(n),
        stream=stream,
    )
    build_cells_ms = _finish_event(start, stop, stream)
    if build_cluster_metadata:
        cluster_context = _build_cluster_metadata(
            total_cells,
            n,
            cluster_size,
            d_cell_particle_counts,
            d_cell_starts,
            stream,
        )
        cluster_total = cluster_context.cluster_total
        cluster_cell = cluster_context.cluster_cell
        cluster_begin = cluster_context.cluster_begin
        cluster_count = cluster_context.cluster_count
        timings_ms = (
            ("build_cells_ms", build_cells_ms),
            ("cluster_ranges_ms", cluster_context.cluster_ranges_ms),
        )
    else:
        cluster_total = 1
        cluster_cell = workspace.cluster_cell
        cluster_begin = workspace.cluster_begin
        cluster_count = workspace.cluster_count
        timings_ms = (("build_cells_ms", build_cells_ms),)
    return FusedCudaNeighborContext(
        n=n,
        x=x,
        y=y,
        z=z,
        h=h,
        lower=lower,
        upper=upper,
        periodic=periodic,
        radius_scale=radius_scale,
        search_radius_cells=np.int32(1),
        cell_counts=cell_counts,
        total_cells=total_cells,
        cluster_total=cluster_total,
        stream=stream,
        sorted_ids=d_sorted_ids,
        cell_starts=d_cell_starts,
        cell_particle_counts=d_cell_particle_counts,
        cluster_cell=cluster_cell,
        cluster_begin=cluster_begin,
        cluster_count=cluster_count,
        timings_ms=timings_ms,
    )


def build_fused_cuda_hbucket_context_with_workspace(
    x: object,
    y: object,
    z: object,
    h: object,
    n: int,
    lower: np.ndarray,
    upper: np.ndarray,
    periodic: np.ndarray,
    radius_scale: np.float32,
    bucket_count: int,
    stream: object,
    workspace: FusedCudaNeighborWorkspace,
    h_reduce_scratch: list[object],
) -> FusedCudaHBucketNeighborContext:
    """Build h-bucket sorted-cell context using reusable device buffers."""
    if workspace.hbucket_h_min_ref is None:
        workspace.hbucket_h_min_ref = reduce_min_float(h, n, stream, h_reduce_scratch)
    h_min = workspace.hbucket_h_min_ref
    return _build_fused_cuda_hbucket_context_from_hmin(
        x,
        y,
        z,
        h,
        n,
        lower,
        upper,
        periodic,
        radius_scale,
        bucket_count,
        stream,
        workspace,
        h_min,
    )


def _build_fused_cuda_hbucket_context_from_hmin(
    x: object,
    y: object,
    z: object,
    h: object,
    n: int,
    lower: np.ndarray,
    upper: np.ndarray,
    periodic: np.ndarray,
    radius_scale: np.float32,
    bucket_count: int,
    stream: object,
    workspace: FusedCudaNeighborWorkspace,
    h_min: np.float32,
) -> FusedCudaHBucketNeighborContext:
    """Build h-bucket sorted-cell context using a host-visible h-min."""
    _ensure_cuda_context()
    assert x.dtype == np.float32
    assert y.dtype == np.float32
    assert z.dtype == np.float32
    assert h.dtype == np.float32
    assert n > 0
    assert lower.dtype == np.float32
    assert upper.dtype == np.float32
    assert periodic.dtype == np.bool_
    assert isinstance(radius_scale, np.float32)
    assert bucket_count > 0
    assert isinstance(h_min, np.float32)
    assert h_min > np.float32(0.0)

    import pycuda.driver as cuda

    cell_counts = cell_counts_from_hmin(lower, upper, h_min, radius_scale)
    total_cells = int(np.prod(cell_counts))
    flat_total = bucket_count * total_cells
    cell_width = ((upper - lower) / cell_counts).astype(np.float32)
    workspace.ensure_hbucket(n, flat_total, bucket_count)
    d_lower = workspace.lower
    d_upper = workspace.upper
    d_cell_bucket_counts = workspace.hbucket_cell_bucket_counts
    d_cell_bucket_starts = workspace.hbucket_cell_bucket_starts
    d_sorted_ids = workspace.hbucket_sorted_ids
    d_particle_cell = workspace.hbucket_particle_cell
    d_particle_bucket = workspace.hbucket_particle_bucket
    d_particle_local_index = workspace.hbucket_particle_local_index
    d_bucket_h_max_bits = workspace.hbucket_bucket_h_max_bits
    d_cell_bucket_h_max_bits = workspace.hbucket_cell_bucket_h_max_bits
    cuda.memcpy_htod_async(_device_ptr(d_lower), np.ascontiguousarray(lower), stream)
    cuda.memcpy_htod_async(_device_ptr(d_upper), np.ascontiguousarray(upper), stream)

    kernels = _module()
    reset_hbucket_metadata = kernels.get_function("fused_reset_hbucket_metadata")
    compute_ids_counts = kernels.get_function("fused_compute_hbucket_ids_counts_xyz")
    scatter_sorted = kernels.get_function("fused_scatter_hbucket_sorted_particles")

    start, stop = _event_pair(stream)
    reset_hbucket_metadata(
        _device_ptr(d_cell_bucket_counts),
        _device_ptr(d_bucket_h_max_bits),
        _device_ptr(d_cell_bucket_h_max_bits),
        np.int32(flat_total),
        np.int32(bucket_count),
        block=(256, 1, 1),
        grid=_grid_size(flat_total),
        stream=stream,
    )
    compute_ids_counts(
        _device_ptr(x),
        _device_ptr(y),
        _device_ptr(z),
        _device_ptr(h),
        np.int32(n),
        _device_ptr(d_lower),
        _device_ptr(d_upper),
        h_min,
        np.int32(bucket_count),
        np.int32(cell_counts[0]),
        np.int32(cell_counts[1]),
        np.int32(cell_counts[2]),
        np.int32(total_cells),
        _device_ptr(d_cell_bucket_counts),
        _device_ptr(d_particle_cell),
        _device_ptr(d_particle_bucket),
        _device_ptr(d_particle_local_index),
        _device_ptr(d_bucket_h_max_bits),
        _device_ptr(d_cell_bucket_h_max_bits),
        block=(256, 1, 1),
        grid=_grid_size(n),
        stream=stream,
    )
    _scan_int32(d_cell_bucket_counts, d_cell_bucket_starts, stream)
    scatter_sorted(
        np.int32(n),
        np.int32(total_cells),
        _device_ptr(d_particle_cell),
        _device_ptr(d_particle_bucket),
        _device_ptr(d_particle_local_index),
        _device_ptr(d_cell_bucket_starts),
        _device_ptr(d_sorted_ids),
        block=(256, 1, 1),
        grid=_grid_size(n),
        stream=stream,
    )
    hbucket_build_ms = _finish_event(start, stop, stream)
    return FusedCudaHBucketNeighborContext(
        n=n,
        x=x,
        y=y,
        z=z,
        h=h,
        lower=lower,
        upper=upper,
        periodic=periodic,
        radius_scale=radius_scale,
        cell_counts=cell_counts,
        total_cells=total_cells,
        bucket_count=bucket_count,
        cell_width=cell_width,
        stream=stream,
        bucket_h_max_bits=d_bucket_h_max_bits,
        cell_bucket_h_max_bits=d_cell_bucket_h_max_bits,
        sorted_ids=d_sorted_ids,
        cell_bucket_starts=d_cell_bucket_starts,
        cell_bucket_counts=d_cell_bucket_counts,
        timings_ms=(("hbucket_build_ms", hbucket_build_ms),),
    )


def count_neighbors_from_context(context: FusedCudaNeighborContext) -> object:
    """Return per-particle neighbor counts from no-CSR context traversal."""
    _ensure_cuda_context()
    import pycuda.gpuarray as gpuarray

    d_neighbor_counts = gpuarray.empty((context.n,), np.int32)
    kernel = _module().get_function("fused_count_neighbors_from_context")
    kernel(
        _device_ptr(context.x),
        _device_ptr(context.y),
        _device_ptr(context.z),
        _device_ptr(context.h),
        np.int32(context.n),
        np.float32(context.lower[0]),
        np.float32(context.upper[0]),
        np.float32(context.lower[1]),
        np.float32(context.upper[1]),
        np.float32(context.lower[2]),
        np.float32(context.upper[2]),
        np.int32(context.periodic[0]),
        np.int32(context.periodic[1]),
        np.int32(context.periodic[2]),
        context.radius_scale,
        np.int32(context.search_radius_cells),
        np.int32(context.cell_counts[0]),
        np.int32(context.cell_counts[1]),
        np.int32(context.cell_counts[2]),
        _device_ptr(context.cell_particle_counts),
        _device_ptr(context.cell_starts),
        _device_ptr(context.sorted_ids),
        _device_ptr(d_neighbor_counts),
        block=(256, 1, 1),
        grid=_grid_size(context.n),
        stream=context.stream,
    )
    return d_neighbor_counts


def count_neighbors_from_hbucket_context(
    context: FusedCudaHBucketNeighborContext,
) -> object:
    """Return per-particle neighbor counts from h-bucket context traversal."""
    _ensure_cuda_context()
    import pycuda.gpuarray as gpuarray

    d_neighbor_counts = gpuarray.empty((context.n,), np.int32)
    kernel = _module().get_function("fused_count_neighbors_from_hbucket_context")
    kernel(
        _device_ptr(context.x),
        _device_ptr(context.y),
        _device_ptr(context.z),
        _device_ptr(context.h),
        np.int32(context.n),
        np.float32(context.lower[0]),
        np.float32(context.upper[0]),
        np.float32(context.lower[1]),
        np.float32(context.upper[1]),
        np.float32(context.lower[2]),
        np.float32(context.upper[2]),
        np.int32(context.periodic[0]),
        np.int32(context.periodic[1]),
        np.int32(context.periodic[2]),
        context.radius_scale,
        np.int32(context.cell_counts[0]),
        np.int32(context.cell_counts[1]),
        np.int32(context.cell_counts[2]),
        np.int32(context.total_cells),
        np.int32(context.bucket_count),
        np.float32(context.cell_width[0]),
        np.float32(context.cell_width[1]),
        np.float32(context.cell_width[2]),
        _device_ptr(context.bucket_h_max_bits),
        _device_ptr(context.cell_bucket_counts),
        _device_ptr(context.cell_bucket_starts),
        _device_ptr(context.sorted_ids),
        _device_ptr(d_neighbor_counts),
        block=(256, 1, 1),
        grid=_grid_size(context.n),
        stream=context.stream,
    )
    return d_neighbor_counts


def count_hbucket_traversal_work_from_context(
    context: FusedCudaHBucketNeighborContext,
) -> tuple[object, object, object]:
    """Return per-particle h-bucket traversal work counters on the device."""
    _ensure_cuda_context()
    import pycuda.gpuarray as gpuarray

    d_visited_cell_counts = gpuarray.empty((context.n,), np.int32)
    d_candidate_counts = gpuarray.empty((context.n,), np.int32)
    d_neighbor_counts = gpuarray.empty((context.n,), np.int32)
    kernel = _module().get_function("fused_count_hbucket_traversal_work")
    kernel(
        _device_ptr(context.x),
        _device_ptr(context.y),
        _device_ptr(context.z),
        _device_ptr(context.h),
        np.int32(context.n),
        np.float32(context.lower[0]),
        np.float32(context.upper[0]),
        np.float32(context.lower[1]),
        np.float32(context.upper[1]),
        np.float32(context.lower[2]),
        np.float32(context.upper[2]),
        np.int32(context.periodic[0]),
        np.int32(context.periodic[1]),
        np.int32(context.periodic[2]),
        context.radius_scale,
        np.int32(context.cell_counts[0]),
        np.int32(context.cell_counts[1]),
        np.int32(context.cell_counts[2]),
        np.int32(context.total_cells),
        np.int32(context.bucket_count),
        np.float32(context.cell_width[0]),
        np.float32(context.cell_width[1]),
        np.float32(context.cell_width[2]),
        _device_ptr(context.bucket_h_max_bits),
        _device_ptr(context.cell_bucket_h_max_bits),
        _device_ptr(context.cell_bucket_counts),
        _device_ptr(context.cell_bucket_starts),
        _device_ptr(context.sorted_ids),
        _device_ptr(d_visited_cell_counts),
        _device_ptr(d_candidate_counts),
        _device_ptr(d_neighbor_counts),
        block=(256, 1, 1),
        grid=_grid_size(context.n),
        stream=context.stream,
    )
    return d_visited_cell_counts, d_candidate_counts, d_neighbor_counts


def reduce_max_float(
    array: object, n: int, stream: object, scratch: list[object]
) -> np.float32:
    """Return the maximum of a device float32 array."""
    return _reduce_float(array, n, stream, scratch, "fused_reduce_max_float")


def reduce_min_float(
    array: object, n: int, stream: object, scratch: list[object]
) -> np.float32:
    """Return the minimum of a device float32 array."""
    return _reduce_float(array, n, stream, scratch, "fused_reduce_min_float")


def wrap_periodic_xyz(
    x: object,
    y: object,
    z: object,
    n: int,
    lower: np.ndarray,
    upper: np.ndarray,
    periodic: np.ndarray,
    stream: object,
) -> None:
    """Wrap device x/y/z coordinates in a minimum-image periodic box."""
    _ensure_cuda_context()
    assert n > 0
    assert lower.dtype == np.float32
    assert upper.dtype == np.float32
    assert periodic.dtype == np.bool_
    kernel = _module().get_function("fused_wrap_periodic_xyz")
    kernel(
        _device_ptr(x),
        _device_ptr(y),
        _device_ptr(z),
        np.int32(n),
        np.float32(lower[0]),
        np.float32(upper[0]),
        np.float32(lower[1]),
        np.float32(upper[1]),
        np.float32(lower[2]),
        np.float32(upper[2]),
        np.int32(periodic[0]),
        np.int32(periodic[1]),
        np.int32(periodic[2]),
        block=(256, 1, 1),
        grid=_grid_size(n),
        stream=stream,
    )


def _reduce_float(
    array: object,
    n: int,
    stream: object,
    scratch: list[object],
    kernel_name: str,
) -> np.float32:
    _ensure_cuda_context()
    assert n > 0
    assert array.dtype == np.float32
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as cuda

    kernel = _module().get_function(kernel_name)
    current = array
    current_n = n
    scratch_index = 0
    while current_n > 1:
        blocks = (current_n + 255) // 256
        if len(scratch) <= scratch_index:
            scratch.append(gpuarray.empty((blocks,), np.float32))
        if scratch[scratch_index].shape[0] < blocks:
            scratch[scratch_index] = gpuarray.empty((blocks,), np.float32)
        out = scratch[scratch_index]
        kernel(
            _device_ptr(current),
            np.int32(current_n),
            _device_ptr(out),
            block=(256, 1, 1),
            grid=_grid_size(current_n),
            stream=stream,
        )
        current = out
        current_n = blocks
        scratch_index = 1 - scratch_index
    host = np.empty((1,), dtype=np.float32)
    cuda.memcpy_dtoh_async(host, _device_ptr(current), stream)
    stream.synchronize()
    return np.float32(host[0])


def create_fused_cuda_convergence_flag(stream: object) -> object:
    """Return a device convergence flag initialized to true."""
    _ensure_cuda_context()
    import pycuda.gpuarray as gpuarray

    flag = gpuarray.empty((1,), np.int32)
    reset_fused_cuda_convergence_flag(flag, stream)
    return flag


def reset_fused_cuda_convergence_flag(flag: object, stream: object) -> None:
    """Reset a device convergence flag to true on the given stream."""
    _ensure_cuda_context()
    assert flag.dtype == np.int32
    kernel = _module().get_function("fused_reset_convergence_flag")
    kernel(
        _device_ptr(flag),
        block=(1, 1, 1),
        grid=(1, 1, 1),
        stream=stream,
    )


def read_fused_cuda_convergence_flag(flag: object, stream: object) -> bool:
    """Read a device convergence flag at a true host boundary."""
    _ensure_cuda_context()
    assert flag.dtype == np.int32
    import pycuda.driver as cuda

    host_flag = np.empty((1,), dtype=np.int32)
    cuda.memcpy_dtoh_async(host_flag, flag.gpudata, stream)
    stream.synchronize()
    return bool(host_flag[0] > np.int32(0))


def cell_counts_from_hmax(
    lower: np.ndarray,
    upper: np.ndarray,
    hmax: np.float32,
    radius_scale: np.float32,
) -> np.ndarray:
    """Return uniform grid cell counts from maximum smoothing length."""
    assert lower.dtype == np.float32
    assert upper.dtype == np.float32
    assert isinstance(hmax, np.float32)
    assert isinstance(radius_scale, np.float32)
    cell_size = np.float32(radius_scale * hmax)
    return cell_counts_from_cell_size(lower, upper, cell_size)


def cell_counts_from_hmin(
    lower: np.ndarray,
    upper: np.ndarray,
    h_min: np.float32,
    radius_scale: np.float32,
) -> np.ndarray:
    """Return h-bucket cell counts from minimum smoothing length."""
    assert lower.dtype == np.float32
    assert upper.dtype == np.float32
    assert isinstance(h_min, np.float32)
    assert isinstance(radius_scale, np.float32)
    cell_size = np.float32(radius_scale * h_min)
    counts = np.ceil((upper - lower) / cell_size).astype(np.int32)
    return np.maximum(counts, np.ones((3,), dtype=np.int32)).astype(np.int32)


def cell_counts_from_cell_size(
    lower: np.ndarray, upper: np.ndarray, cell_size: np.float32
) -> np.ndarray:
    """Return uniform grid cell counts from a host-side binning cell size."""
    assert lower.dtype == np.float32
    assert upper.dtype == np.float32
    assert isinstance(cell_size, np.float32)
    counts = np.floor((upper - lower) / cell_size).astype(np.int32)
    return np.maximum(counts, np.ones((3,), dtype=np.int32)).astype(np.int32)


@dataclass(frozen=True)
class _ClusterMetadata:
    cluster_total: int
    cluster_cell: object
    cluster_begin: object
    cluster_count: object
    cluster_ranges_ms: float


def _build_cluster_metadata(
    total_cells: int,
    n: int,
    cluster_size: int,
    cell_counts: object,
    cell_starts: object,
    stream: object,
) -> _ClusterMetadata:
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as cuda

    d_cell_cluster_count = gpuarray.empty((total_cells,), np.int32)
    d_cell_cluster_start = gpuarray.empty((total_cells,), np.int32)
    d_cluster_total = gpuarray.empty((1,), np.int32)
    d_cluster_cell = gpuarray.empty((n,), np.int32)
    d_cluster_begin = gpuarray.empty((n,), np.int32)
    d_cluster_count = gpuarray.empty((n,), np.int32)

    kernels = _module()
    reset_int = kernels.get_function("fused_reset_int")
    cluster_counts = kernels.get_function("fused_cluster_counts_from_cells")
    fill_clusters = kernels.get_function("fused_fill_destination_clusters")
    last_cluster_total = kernels.get_function("fused_last_cluster_total")

    start, stop = _event_pair(stream)
    reset_int(
        _device_ptr(d_cluster_count),
        np.int32(n),
        block=(256, 1, 1),
        grid=_grid_size(n),
        stream=stream,
    )
    cluster_counts(
        np.int32(total_cells),
        _device_ptr(cell_counts),
        np.int32(cluster_size),
        _device_ptr(d_cell_cluster_count),
        block=(256, 1, 1),
        grid=_grid_size(total_cells),
        stream=stream,
    )
    _scan_int32(d_cell_cluster_count, d_cell_cluster_start, stream)
    fill_clusters(
        np.int32(total_cells),
        np.int32(cluster_size),
        _device_ptr(cell_counts),
        _device_ptr(cell_starts),
        _device_ptr(d_cell_cluster_start),
        _device_ptr(d_cluster_cell),
        _device_ptr(d_cluster_begin),
        _device_ptr(d_cluster_count),
        block=(256, 1, 1),
        grid=_grid_size(total_cells),
        stream=stream,
    )
    last_cluster_total(
        np.int32(total_cells),
        _device_ptr(d_cell_cluster_start),
        _device_ptr(d_cell_cluster_count),
        _device_ptr(d_cluster_total),
        block=(1, 1, 1),
        grid=(1, 1, 1),
        stream=stream,
    )
    cluster_ranges_ms = _finish_event(start, stop, stream)
    host_cluster_total = np.empty((1,), dtype=np.int32)
    cuda.memcpy_dtoh_async(host_cluster_total, _device_ptr(d_cluster_total), stream)
    stream.synchronize()
    cluster_total = int(host_cluster_total[0])
    assert cluster_total > 0
    return _ClusterMetadata(
        cluster_total=cluster_total,
        cluster_cell=d_cluster_cell,
        cluster_begin=d_cluster_begin,
        cluster_count=d_cluster_count,
        cluster_ranges_ms=cluster_ranges_ms,
    )


def _ensure_cuda_context() -> None:
    import pycuda.autoinit
    import pycuda.driver as cuda

    cuda.init()
    assert cuda.Device.count() > 0
    assert pycuda.autoinit.context is not None


def _ensure_gpu_array(array: object, size: int, dtype: object) -> object:
    import pycuda.gpuarray as gpuarray

    if array is None:
        return gpuarray.empty((size,), dtype)
    if array.shape[0] < size:
        return gpuarray.empty((size,), dtype)
    return array


def _grid_size(n: int) -> tuple[int, int, int]:
    return ((n + 255) // 256, 1, 1)


def _device_ptr(array: object) -> object:
    gpudata = array.gpudata
    if isinstance(gpudata, int):
        return np.uintp(gpudata)
    return gpudata


def _event_pair(stream: object) -> tuple[object, object]:
    if not _profile_cuda_events_enabled():
        return None, None
    import pycuda.driver as cuda

    start = cuda.Event()
    stop = cuda.Event()
    start.record(stream)
    return start, stop


def _finish_event(start: object, stop: object, stream: object) -> float:
    if start is None:
        return 0.0
    stop.record(stream)
    stop.synchronize()
    return float(stop.time_since(start))


def _profile_cuda_events_enabled() -> bool:
    value = os.environ.get("PYSPH_PROFILE_CUDA_EVENTS")
    if value is None:
        return False
    assert value == "1"
    return True


def _scan_int32(input_array: object, output_array: object, stream: object) -> None:
    scan = _scan_kernel()
    scan(input_array, output_array, stream=stream)


def _scan_kernel() -> object:
    global _SCAN_KERNEL
    from pycuda.scan import ExclusiveScanKernel

    if _SCAN_KERNEL is None:
        _SCAN_KERNEL = ExclusiveScanKernel(np.int32, "a+b", neutral="0")
    return _SCAN_KERNEL


@cache
def _module() -> object:
    from pycuda.compiler import SourceModule

    return SourceModule(CUDA_SOURCE, no_extern_c=True)


def minimum_image_delta(
    a: np.ndarray,
    b: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    periodic: np.ndarray,
) -> np.ndarray:
    """Return PySPH-style destination-source minimum-image displacement."""
    assert a.dtype == np.float32
    assert b.dtype == np.float32
    assert lower.dtype == np.float32
    assert upper.dtype == np.float32
    assert periodic.dtype == np.bool_
    delta = (a - b).astype(np.float32)
    lengths = (upper - lower).astype(np.float32)
    half_lengths = np.float32(0.5) * lengths
    wrap_high = np.logical_and(periodic, delta > half_lengths).astype(np.float32)
    wrap_low = np.logical_and(periodic, delta < -half_lengths).astype(np.float32)
    return (delta - lengths * wrap_high + lengths * wrap_low).astype(np.float32)


def brute_force_neighbor_indices(
    xyz: np.ndarray,
    h: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    periodic: np.ndarray,
    radius_scale: np.float32,
    sample: np.int32,
) -> np.ndarray:
    """Return strict-cutoff neighbor indices for a destination particle."""
    assert xyz.dtype == np.float32
    assert h.dtype == np.float32
    assert lower.dtype == np.float32
    assert upper.dtype == np.float32
    assert periodic.dtype == np.bool_
    assert isinstance(radius_scale, np.float32)
    assert isinstance(sample, np.int32)
    delta = _minimum_image_deltas(xyz[int(sample)], xyz, lower, upper, periodic)
    dist2 = np.sum(delta * delta, axis=1).astype(np.float32)
    support = radius_scale * np.maximum(h[int(sample)], h)
    return np.flatnonzero(dist2 < support * support).astype(np.int32)


def _minimum_image_deltas(
    a: np.ndarray,
    b: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    periodic: np.ndarray,
) -> np.ndarray:
    """Return vectorized minimum-image displacements from one destination."""
    delta = (a[None, :] - b).astype(np.float32)
    lengths = (upper - lower).astype(np.float32)
    half_lengths = np.float32(0.5) * lengths
    wrap_high = np.logical_and(periodic[None, :], delta > half_lengths[None, :]).astype(
        np.float32
    )
    wrap_low = np.logical_and(periodic[None, :], delta < -half_lengths[None, :]).astype(
        np.float32
    )
    return (delta - lengths[None, :] * wrap_high + lengths[None, :] * wrap_low).astype(
        np.float32
    )
