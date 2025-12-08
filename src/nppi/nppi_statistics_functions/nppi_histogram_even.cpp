#include "npp.h"
#include "../../npp_version_compat.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for implementation functions
extern "C" {
NppStatus nppiHistogramEven_16u_C4R_Ctx_impl(const Npp16u *pSrc, int nSrcStep,
                                             NppiSize oSizeROI,
                                             Npp32s *pHist[4],
                                             int nLevels[4],
                                             Npp32s nLowerLevel[4],
                                             Npp32s nUpperLevel[4],
                                             Npp8u *pDeviceBuffer,
                                             NppStreamContext nppStreamCtx);

NppStatus nppiHistogramEven_16s_C4R_Ctx_impl(const Npp16s *pSrc, int nSrcStep,
                                             NppiSize oSizeROI,
                                             Npp32s *pHist[4],
                                             int nLevels[4],
                                             Npp32s nLowerLevel[4],
                                             Npp32s nUpperLevel[4],
                                             Npp8u *pDeviceBuffer,
                                             NppStreamContext nppStreamCtx);

NppStatus nppiHistogramEvenGetBufferSize_16u_C4R_Ctx_impl(NppiSize oSizeROI,
                                                          int nLevels[4],
                                                          size_t *hpBufferSize);

NppStatus nppiHistogramEvenGetBufferSize_16s_C4R_Ctx_impl(NppiSize oSizeROI,
                                                          int nLevels[4],
                                                          size_t *hpBufferSize);
}

// Input validation helper
static inline NppStatus validateHistogramEvenInputs(const void *pSrc,
                                                     int nSrcStep,
                                                     NppiSize oSizeROI,
                                                     Npp32s *pHist[4],
                                                     int nLevels[4],
                                                     Npp32s nLowerLevel[4],
                                                     Npp32s nUpperLevel[4],
                                                     const Npp8u *pDeviceBuffer) {
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (!pSrc || !pHist || !nLevels || !nLowerLevel || !nUpperLevel || !pDeviceBuffer) {
    return NPP_NULL_POINTER_ERROR;
  }

  // Validate individual channel histogram pointers
  for (int c = 0; c < 4; c++) {
    if (!pHist[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
  }

  // Validate range for each channel
  for (int c = 0; c < 4; c++) {
    if (nLowerLevel[c] >= nUpperLevel[c]) {
      return NPP_RANGE_ERROR;
    }
  }

  return NPP_SUCCESS;
}

// ============================================================================
// CUDA SDK 12.8+ versions (size_t)
// ============================================================================

// nppiHistogramEvenGetBufferSize_16u_C4R_Ctx - CUDA SDK 12.8+
NppStatus nppiHistogramEvenGetBufferSize_16u_C4R_Ctx(NppiSize oSizeROI,
                                                     int nLevels[4],
                                                     size_t *hpBufferSize,
                                                     NppStreamContext /*nppStreamCtx*/) {
  if (!hpBufferSize || !nLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  return nppiHistogramEvenGetBufferSize_16u_C4R_Ctx_impl(oSizeROI, nLevels, hpBufferSize);
}

// nppiHistogramEvenGetBufferSize_16u_C4R - CUDA SDK 12.8+
NppStatus nppiHistogramEvenGetBufferSize_16u_C4R(NppiSize oSizeROI,
                                                 int nLevels[4],
                                                 size_t *hpBufferSize) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiHistogramEvenGetBufferSize_16u_C4R_Ctx(oSizeROI, nLevels, hpBufferSize, nppStreamCtx);
}

// nppiHistogramEvenGetBufferSize_16s_C4R_Ctx - CUDA SDK 12.8+
NppStatus nppiHistogramEvenGetBufferSize_16s_C4R_Ctx(NppiSize oSizeROI,
                                                     int nLevels[4],
                                                     size_t *hpBufferSize,
                                                     NppStreamContext /*nppStreamCtx*/) {
  if (!hpBufferSize || !nLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  return nppiHistogramEvenGetBufferSize_16s_C4R_Ctx_impl(oSizeROI, nLevels, hpBufferSize);
}

// nppiHistogramEvenGetBufferSize_16s_C4R - CUDA SDK 12.8+
NppStatus nppiHistogramEvenGetBufferSize_16s_C4R(NppiSize oSizeROI,
                                                 int nLevels[4],
                                                 size_t *hpBufferSize) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiHistogramEvenGetBufferSize_16s_C4R_Ctx(oSizeROI, nLevels, hpBufferSize, nppStreamCtx);
}

// ============================================================================
// CUDA SDK < 12.8 versions (int) - Function overloads
// ============================================================================

// nppiHistogramEvenGetBufferSize_16u_C4R_Ctx - CUDA SDK < 12.8
NppStatus nppiHistogramEvenGetBufferSize_16u_C4R_Ctx(NppiSize oSizeROI,
                                                     int nLevels[4],
                                                     int *hpBufferSize,
                                                     NppStreamContext nppStreamCtx) {
  size_t size = 0;
  NppStatus ret = nppiHistogramEvenGetBufferSize_16u_C4R_Ctx(oSizeROI, nLevels, &size, nppStreamCtx);
  if (ret == NPP_SUCCESS) {
    *hpBufferSize = static_cast<int>(size);
  }
  return ret;
}

// nppiHistogramEvenGetBufferSize_16u_C4R - CUDA SDK < 12.8
NppStatus nppiHistogramEvenGetBufferSize_16u_C4R(NppiSize oSizeROI,
                                                 int nLevels[4],
                                                 int *hpBufferSize) {
  size_t size = 0;
  NppStatus ret = nppiHistogramEvenGetBufferSize_16u_C4R(oSizeROI, nLevels, &size);
  if (ret == NPP_SUCCESS) {
    *hpBufferSize = static_cast<int>(size);
  }
  return ret;
}

// nppiHistogramEvenGetBufferSize_16s_C4R_Ctx - CUDA SDK < 12.8
NppStatus nppiHistogramEvenGetBufferSize_16s_C4R_Ctx(NppiSize oSizeROI,
                                                     int nLevels[4],
                                                     int *hpBufferSize,
                                                     NppStreamContext nppStreamCtx) {
  size_t size = 0;
  NppStatus ret = nppiHistogramEvenGetBufferSize_16s_C4R_Ctx(oSizeROI, nLevels, &size, nppStreamCtx);
  if (ret == NPP_SUCCESS) {
    *hpBufferSize = static_cast<int>(size);
  }
  return ret;
}

// nppiHistogramEvenGetBufferSize_16s_C4R - CUDA SDK < 12.8
NppStatus nppiHistogramEvenGetBufferSize_16s_C4R(NppiSize oSizeROI,
                                                 int nLevels[4],
                                                 int *hpBufferSize) {
  size_t size = 0;
  NppStatus ret = nppiHistogramEvenGetBufferSize_16s_C4R(oSizeROI, nLevels, &size);
  if (ret == NPP_SUCCESS) {
    *hpBufferSize = static_cast<int>(size);
  }
  return ret;
}

// ============================================================================
// Histogram computation functions (version-independent)
// ============================================================================

// nppiHistogramEven_16u_C4R_Ctx
NppStatus nppiHistogramEven_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                        NppiSize oSizeROI,
                                        Npp32s *pHist[4],
                                        int nLevels[4],
                                        Npp32s nLowerLevel[4],
                                        Npp32s nUpperLevel[4],
                                        Npp8u *pDeviceBuffer,
                                        NppStreamContext nppStreamCtx) {
  NppStatus status = validateHistogramEvenInputs(pSrc, nSrcStep, oSizeROI, pHist,
                                                  nLevels, nLowerLevel, nUpperLevel,
                                                  pDeviceBuffer);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiHistogramEven_16u_C4R_Ctx_impl(pSrc, nSrcStep, oSizeROI, pHist,
                                            nLevels, nLowerLevel, nUpperLevel,
                                            pDeviceBuffer, nppStreamCtx);
}

// nppiHistogramEven_16u_C4R
NppStatus nppiHistogramEven_16u_C4R(const Npp16u *pSrc, int nSrcStep,
                                    NppiSize oSizeROI,
                                    Npp32s *pHist[4],
                                    int nLevels[4],
                                    Npp32s nLowerLevel[4],
                                    Npp32s nUpperLevel[4],
                                    Npp8u *pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiHistogramEven_16u_C4R_Ctx(pSrc, nSrcStep, oSizeROI, pHist,
                                       nLevels, nLowerLevel, nUpperLevel,
                                       pDeviceBuffer, nppStreamCtx);
}

// nppiHistogramEven_16s_C4R_Ctx
NppStatus nppiHistogramEven_16s_C4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                        NppiSize oSizeROI,
                                        Npp32s *pHist[4],
                                        int nLevels[4],
                                        Npp32s nLowerLevel[4],
                                        Npp32s nUpperLevel[4],
                                        Npp8u *pDeviceBuffer,
                                        NppStreamContext nppStreamCtx) {
  NppStatus status = validateHistogramEvenInputs(pSrc, nSrcStep, oSizeROI, pHist,
                                                  nLevels, nLowerLevel, nUpperLevel,
                                                  pDeviceBuffer);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiHistogramEven_16s_C4R_Ctx_impl(pSrc, nSrcStep, oSizeROI, pHist,
                                            nLevels, nLowerLevel, nUpperLevel,
                                            pDeviceBuffer, nppStreamCtx);
}

// nppiHistogramEven_16s_C4R
NppStatus nppiHistogramEven_16s_C4R(const Npp16s *pSrc, int nSrcStep,
                                    NppiSize oSizeROI,
                                    Npp32s *pHist[4],
                                    int nLevels[4],
                                    Npp32s nLowerLevel[4],
                                    Npp32s nUpperLevel[4],
                                    Npp8u *pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiHistogramEven_16s_C4R_Ctx(pSrc, nSrcStep, oSizeROI, pHist,
                                       nLevels, nLowerLevel, nUpperLevel,
                                       pDeviceBuffer, nppStreamCtx);
}
