#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Multi-channel histogram kernels - 16u C4R
__global__ void nppiHistogramEven_16u_C4R_kernel_shared(const Npp16u *pSrc, int nSrcStep, int width, int height,
                                                        Npp32s *pHist0, Npp32s *pHist1, Npp32s *pHist2, Npp32s *pHist3,
                                                        int nLevels0, int nLevels1, int nLevels2, int nLevels3,
                                                        Npp32s nLowerLevel0, Npp32s nLowerLevel1, Npp32s nLowerLevel2,
                                                        Npp32s nLowerLevel3, Npp32s nUpperLevel0, Npp32s nUpperLevel1,
                                                        Npp32s nUpperLevel2, Npp32s nUpperLevel3) {
  extern __shared__ int shared_hist[];

  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int threads_per_block = blockDim.x * blockDim.y;

  // Calculate total bins needed across all channels
  int totalBins = (nLevels0 - 1) + (nLevels1 - 1) + (nLevels2 - 1) + (nLevels3 - 1);

  // Initialize shared memory for all channels
  for (int i = tid; i < totalBins; i += threads_per_block) {
    shared_hist[i] = 0;
  }
  __syncthreads();

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp16u *src_row = (const Npp16u *)((const char *)pSrc + y * nSrcStep);

    // Process channel 0
    int pixel_value = src_row[x * 4 + 0];
    if (pixel_value >= nLowerLevel0 && pixel_value < nUpperLevel0) {
      int range = nUpperLevel0 - nLowerLevel0;
      long long bin = ((long long)(pixel_value - nLowerLevel0) * (nLevels0 - 1)) / range;
      if (bin >= nLevels0 - 1) bin = nLevels0 - 2;
      atomicAdd(&shared_hist[bin], 1);
    }

    // Process channel 1
    pixel_value = src_row[x * 4 + 1];
    if (pixel_value >= nLowerLevel1 && pixel_value < nUpperLevel1) {
      int range = nUpperLevel1 - nLowerLevel1;
      long long bin = ((long long)(pixel_value - nLowerLevel1) * (nLevels1 - 1)) / range;
      if (bin >= nLevels1 - 1) bin = nLevels1 - 2;
      atomicAdd(&shared_hist[(nLevels0 - 1) + bin], 1);
    }

    // Process channel 2
    pixel_value = src_row[x * 4 + 2];
    if (pixel_value >= nLowerLevel2 && pixel_value < nUpperLevel2) {
      int range = nUpperLevel2 - nLowerLevel2;
      long long bin = ((long long)(pixel_value - nLowerLevel2) * (nLevels2 - 1)) / range;
      if (bin >= nLevels2 - 1) bin = nLevels2 - 2;
      atomicAdd(&shared_hist[(nLevels0 - 1) + (nLevels1 - 1) + bin], 1);
    }

    // Process channel 3
    pixel_value = src_row[x * 4 + 3];
    if (pixel_value >= nLowerLevel3 && pixel_value < nUpperLevel3) {
      int range = nUpperLevel3 - nLowerLevel3;
      long long bin = ((long long)(pixel_value - nLowerLevel3) * (nLevels3 - 1)) / range;
      if (bin >= nLevels3 - 1) bin = nLevels3 - 2;
      atomicAdd(&shared_hist[(nLevels0 - 1) + (nLevels1 - 1) + (nLevels2 - 1) + bin], 1);
    }
  }

  __syncthreads();

  // Copy shared histograms to global memory
  // Channel 0
  for (int i = tid; i < nLevels0 - 1; i += threads_per_block) {
    atomicAdd(&pHist0[i], shared_hist[i]);
  }
  // Channel 1
  for (int i = tid; i < nLevels1 - 1; i += threads_per_block) {
    atomicAdd(&pHist1[i], shared_hist[(nLevels0 - 1) + i]);
  }
  // Channel 2
  for (int i = tid; i < nLevels2 - 1; i += threads_per_block) {
    atomicAdd(&pHist2[i], shared_hist[(nLevels0 - 1) + (nLevels1 - 1) + i]);
  }
  // Channel 3
  for (int i = tid; i < nLevels3 - 1; i += threads_per_block) {
    atomicAdd(&pHist3[i], shared_hist[(nLevels0 - 1) + (nLevels1 - 1) + (nLevels2 - 1) + i]);
  }
}

__global__ void nppiHistogramEven_16u_C4R_kernel_global(const Npp16u *pSrc, int nSrcStep, int width, int height,
                                                        Npp32s *pHist0, Npp32s *pHist1, Npp32s *pHist2, Npp32s *pHist3,
                                                        int nLevels0, int nLevels1, int nLevels2, int nLevels3,
                                                        Npp32s nLowerLevel0, Npp32s nLowerLevel1, Npp32s nLowerLevel2,
                                                        Npp32s nLowerLevel3, Npp32s nUpperLevel0, Npp32s nUpperLevel1,
                                                        Npp32s nUpperLevel2, Npp32s nUpperLevel3) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp16u *src_row = (const Npp16u *)((const char *)pSrc + y * nSrcStep);

    // Process channel 0
    int pixel_value = src_row[x * 4 + 0];
    if (pixel_value >= nLowerLevel0 && pixel_value < nUpperLevel0) {
      int range = nUpperLevel0 - nLowerLevel0;
      long long bin = ((long long)(pixel_value - nLowerLevel0) * (nLevels0 - 1)) / range;
      if (bin >= nLevels0 - 1) bin = nLevels0 - 2;
      atomicAdd(&pHist0[bin], 1);
    }

    // Process channel 1
    pixel_value = src_row[x * 4 + 1];
    if (pixel_value >= nLowerLevel1 && pixel_value < nUpperLevel1) {
      int range = nUpperLevel1 - nLowerLevel1;
      long long bin = ((long long)(pixel_value - nLowerLevel1) * (nLevels1 - 1)) / range;
      if (bin >= nLevels1 - 1) bin = nLevels1 - 2;
      atomicAdd(&pHist1[bin], 1);
    }

    // Process channel 2
    pixel_value = src_row[x * 4 + 2];
    if (pixel_value >= nLowerLevel2 && pixel_value < nUpperLevel2) {
      int range = nUpperLevel2 - nLowerLevel2;
      long long bin = ((long long)(pixel_value - nLowerLevel2) * (nLevels2 - 1)) / range;
      if (bin >= nLevels2 - 1) bin = nLevels2 - 2;
      atomicAdd(&pHist2[bin], 1);
    }

    // Process channel 3
    pixel_value = src_row[x * 4 + 3];
    if (pixel_value >= nLowerLevel3 && pixel_value < nUpperLevel3) {
      int range = nUpperLevel3 - nLowerLevel3;
      long long bin = ((long long)(pixel_value - nLowerLevel3) * (nLevels3 - 1)) / range;
      if (bin >= nLevels3 - 1) bin = nLevels3 - 2;
      atomicAdd(&pHist3[bin], 1);
    }
  }
}

// Multi-channel histogram kernels - 16s C4R
__global__ void nppiHistogramEven_16s_C4R_kernel_shared(const Npp16s *pSrc, int nSrcStep, int width, int height,
                                                        Npp32s *pHist0, Npp32s *pHist1, Npp32s *pHist2, Npp32s *pHist3,
                                                        int nLevels0, int nLevels1, int nLevels2, int nLevels3,
                                                        Npp32s nLowerLevel0, Npp32s nLowerLevel1, Npp32s nLowerLevel2,
                                                        Npp32s nLowerLevel3, Npp32s nUpperLevel0, Npp32s nUpperLevel1,
                                                        Npp32s nUpperLevel2, Npp32s nUpperLevel3) {
  extern __shared__ int shared_hist[];

  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int threads_per_block = blockDim.x * blockDim.y;

  // Calculate total bins needed across all channels
  int totalBins = (nLevels0 - 1) + (nLevels1 - 1) + (nLevels2 - 1) + (nLevels3 - 1);

  // Initialize shared memory for all channels
  for (int i = tid; i < totalBins; i += threads_per_block) {
    shared_hist[i] = 0;
  }
  __syncthreads();

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp16s *src_row = (const Npp16s *)((const char *)pSrc + y * nSrcStep);

    // Process channel 0
    int pixel_value = src_row[x * 4 + 0];
    if (pixel_value >= nLowerLevel0 && pixel_value < nUpperLevel0) {
      int range = nUpperLevel0 - nLowerLevel0;
      long long bin = ((long long)(pixel_value - nLowerLevel0) * (nLevels0 - 1)) / range;
      if (bin >= nLevels0 - 1) bin = nLevels0 - 2;
      atomicAdd(&shared_hist[bin], 1);
    }

    // Process channel 1
    pixel_value = src_row[x * 4 + 1];
    if (pixel_value >= nLowerLevel1 && pixel_value < nUpperLevel1) {
      int range = nUpperLevel1 - nLowerLevel1;
      long long bin = ((long long)(pixel_value - nLowerLevel1) * (nLevels1 - 1)) / range;
      if (bin >= nLevels1 - 1) bin = nLevels1 - 2;
      atomicAdd(&shared_hist[(nLevels0 - 1) + bin], 1);
    }

    // Process channel 2
    pixel_value = src_row[x * 4 + 2];
    if (pixel_value >= nLowerLevel2 && pixel_value < nUpperLevel2) {
      int range = nUpperLevel2 - nLowerLevel2;
      long long bin = ((long long)(pixel_value - nLowerLevel2) * (nLevels2 - 1)) / range;
      if (bin >= nLevels2 - 1) bin = nLevels2 - 2;
      atomicAdd(&shared_hist[(nLevels0 - 1) + (nLevels1 - 1) + bin], 1);
    }

    // Process channel 3
    pixel_value = src_row[x * 4 + 3];
    if (pixel_value >= nLowerLevel3 && pixel_value < nUpperLevel3) {
      int range = nUpperLevel3 - nLowerLevel3;
      long long bin = ((long long)(pixel_value - nLowerLevel3) * (nLevels3 - 1)) / range;
      if (bin >= nLevels3 - 1) bin = nLevels3 - 2;
      atomicAdd(&shared_hist[(nLevels0 - 1) + (nLevels1 - 1) + (nLevels2 - 1) + bin], 1);
    }
  }

  __syncthreads();

  // Copy shared histograms to global memory
  // Channel 0
  for (int i = tid; i < nLevels0 - 1; i += threads_per_block) {
    atomicAdd(&pHist0[i], shared_hist[i]);
  }
  // Channel 1
  for (int i = tid; i < nLevels1 - 1; i += threads_per_block) {
    atomicAdd(&pHist1[i], shared_hist[(nLevels0 - 1) + i]);
  }
  // Channel 2
  for (int i = tid; i < nLevels2 - 1; i += threads_per_block) {
    atomicAdd(&pHist2[i], shared_hist[(nLevels0 - 1) + (nLevels1 - 1) + i]);
  }
  // Channel 3
  for (int i = tid; i < nLevels3 - 1; i += threads_per_block) {
    atomicAdd(&pHist3[i], shared_hist[(nLevels0 - 1) + (nLevels1 - 1) + (nLevels2 - 1) + i]);
  }
}

__global__ void nppiHistogramEven_16s_C4R_kernel_global(const Npp16s *pSrc, int nSrcStep, int width, int height,
                                                        Npp32s *pHist0, Npp32s *pHist1, Npp32s *pHist2, Npp32s *pHist3,
                                                        int nLevels0, int nLevels1, int nLevels2, int nLevels3,
                                                        Npp32s nLowerLevel0, Npp32s nLowerLevel1, Npp32s nLowerLevel2,
                                                        Npp32s nLowerLevel3, Npp32s nUpperLevel0, Npp32s nUpperLevel1,
                                                        Npp32s nUpperLevel2, Npp32s nUpperLevel3) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp16s *src_row = (const Npp16s *)((const char *)pSrc + y * nSrcStep);

    // Process channel 0
    int pixel_value = src_row[x * 4 + 0];
    if (pixel_value >= nLowerLevel0 && pixel_value < nUpperLevel0) {
      int range = nUpperLevel0 - nLowerLevel0;
      long long bin = ((long long)(pixel_value - nLowerLevel0) * (nLevels0 - 1)) / range;
      if (bin >= nLevels0 - 1) bin = nLevels0 - 2;
      atomicAdd(&pHist0[bin], 1);
    }

    // Process channel 1
    pixel_value = src_row[x * 4 + 1];
    if (pixel_value >= nLowerLevel1 && pixel_value < nUpperLevel1) {
      int range = nUpperLevel1 - nLowerLevel1;
      long long bin = ((long long)(pixel_value - nLowerLevel1) * (nLevels1 - 1)) / range;
      if (bin >= nLevels1 - 1) bin = nLevels1 - 2;
      atomicAdd(&pHist1[bin], 1);
    }

    // Process channel 2
    pixel_value = src_row[x * 4 + 2];
    if (pixel_value >= nLowerLevel2 && pixel_value < nUpperLevel2) {
      int range = nUpperLevel2 - nLowerLevel2;
      long long bin = ((long long)(pixel_value - nLowerLevel2) * (nLevels2 - 1)) / range;
      if (bin >= nLevels2 - 1) bin = nLevels2 - 2;
      atomicAdd(&pHist2[bin], 1);
    }

    // Process channel 3
    pixel_value = src_row[x * 4 + 3];
    if (pixel_value >= nLowerLevel3 && pixel_value < nUpperLevel3) {
      int range = nUpperLevel3 - nLowerLevel3;
      long long bin = ((long long)(pixel_value - nLowerLevel3) * (nLevels3 - 1)) / range;
      if (bin >= nLevels3 - 1) bin = nLevels3 - 2;
      atomicAdd(&pHist3[bin], 1);
    }
  }
}

extern "C" {

// Get buffer size for histogram computation

NppStatus nppiHistogramEvenGetBufferSize_16u_C4R_Ctx_impl(NppiSize oSizeROI, int nLevels[4], size_t *hpBufferSize) {
  // Calculate required buffer size for 4-channel histogram
  size_t totalHistogramSize = 0;
  for (int c = 0; c < 4; c++) {
    totalHistogramSize += (size_t)(nLevels[c] - 1) * sizeof(Npp32s);
  }

  // Buffer size includes space for all channel histograms and some temporary workspace
  *hpBufferSize = totalHistogramSize + 1024;

  return NPP_SUCCESS;
}

NppStatus nppiHistogramEvenGetBufferSize_16s_C4R_Ctx_impl(NppiSize oSizeROI, int nLevels[4], size_t *hpBufferSize) {
  // Calculate required buffer size for 4-channel histogram
  size_t totalHistogramSize = 0;
  for (int c = 0; c < 4; c++) {
    totalHistogramSize += (size_t)(nLevels[c] - 1) * sizeof(Npp32s);
  }

  // Buffer size includes space for all channel histograms and some temporary workspace
  *hpBufferSize = totalHistogramSize + 1024;

  return NPP_SUCCESS;
}

// Multi-channel histogram computation for 16u C4R
NppStatus nppiHistogramEven_16u_C4R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[4],
                                             int nLevels[4], Npp32s nLowerLevel[4], Npp32s nUpperLevel[4],
                                             Npp8u *pDeviceBuffer, NppStreamContext nppStreamCtx) {
  // Clear all channel histograms
  for (int c = 0; c < 4; c++) {
    cudaMemset(pHist[c], 0, (nLevels[c] - 1) * sizeof(Npp32s));
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  size_t imageSize = (size_t)oSizeROI.width * oSizeROI.height;

  // Calculate total bins across all channels for shared memory optimization
  int totalBins = (nLevels[0] - 1) + (nLevels[1] - 1) + (nLevels[2] - 1) + (nLevels[3] - 1);

  if (totalBins <= 512 && imageSize < 1024 * 1024) {
    // Use shared memory kernel for efficiency
    size_t sharedMemSize = totalBins * sizeof(int);

    nppiHistogramEven_16u_C4R_kernel_shared<<<gridSize, blockSize, sharedMemSize, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, pHist[0], pHist[1], pHist[2], pHist[3], nLevels[0], nLevels[1],
        nLevels[2], nLevels[3], nLowerLevel[0], nLowerLevel[1], nLowerLevel[2], nLowerLevel[3], nUpperLevel[0],
        nUpperLevel[1], nUpperLevel[2], nUpperLevel[3]);
  } else {
    // Use global memory kernel for large histograms
    nppiHistogramEven_16u_C4R_kernel_global<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, pHist[0], pHist[1], pHist[2], pHist[3], nLevels[0], nLevels[1],
        nLevels[2], nLevels[3], nLowerLevel[0], nLowerLevel[1], nLowerLevel[2], nLowerLevel[3], nUpperLevel[0],
        nUpperLevel[1], nUpperLevel[2], nUpperLevel[3]);
  }

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// Multi-channel histogram computation for 16s C4R
NppStatus nppiHistogramEven_16s_C4R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[4],
                                             int nLevels[4], Npp32s nLowerLevel[4], Npp32s nUpperLevel[4],
                                             Npp8u *pDeviceBuffer, NppStreamContext nppStreamCtx) {
  // Clear all channel histograms
  for (int c = 0; c < 4; c++) {
    cudaMemset(pHist[c], 0, (nLevels[c] - 1) * sizeof(Npp32s));
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  size_t imageSize = (size_t)oSizeROI.width * oSizeROI.height;

  // Calculate total bins across all channels for shared memory optimization
  int totalBins = (nLevels[0] - 1) + (nLevels[1] - 1) + (nLevels[2] - 1) + (nLevels[3] - 1);

  if (totalBins <= 512 && imageSize < 1024 * 1024) {
    // Use shared memory kernel for efficiency
    size_t sharedMemSize = totalBins * sizeof(int);

    nppiHistogramEven_16s_C4R_kernel_shared<<<gridSize, blockSize, sharedMemSize, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, pHist[0], pHist[1], pHist[2], pHist[3], nLevels[0], nLevels[1],
        nLevels[2], nLevels[3], nLowerLevel[0], nLowerLevel[1], nLowerLevel[2], nLowerLevel[3], nUpperLevel[0],
        nUpperLevel[1], nUpperLevel[2], nUpperLevel[3]);
  } else {
    // Use global memory kernel for large histograms
    nppiHistogramEven_16s_C4R_kernel_global<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSizeROI.width, oSizeROI.height, pHist[0], pHist[1], pHist[2], pHist[3], nLevels[0], nLevels[1],
        nLevels[2], nLevels[3], nLowerLevel[0], nLowerLevel[1], nLowerLevel[2], nLowerLevel[3], nUpperLevel[0],
        nUpperLevel[1], nUpperLevel[2], nUpperLevel[3]);
  }

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}
}
