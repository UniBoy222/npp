#include "npp.h"
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
                                                          int *hpBufferSize);

NppStatus nppiHistogramEvenGetBufferSize_16s_C4R_Ctx_impl(NppiSize oSizeROI,
                                                          int nLevels[4],
                                                          int *hpBufferSize);
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
// nppiHistogramEvenGetBufferSize_16u_C4R - Get buffer size for 16u C4R
// ============================================================================

NppStatus nppiHistogramEvenGetBufferSize_16u_C4R_Ctx(NppiSize oSizeROI,
                                                     int nLevels[4],
                                                     int *hpBufferSize,
                                                     NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx; // Stream context not used for buffer size calculation

  if (!hpBufferSize || !nLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  return nppiHistogramEvenGetBufferSize_16u_C4R_Ctx_impl(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramEvenGetBufferSize_16u_C4R(NppiSize oSizeROI,
                                                 int nLevels[4],
                                                 int *hpBufferSize) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiHistogramEvenGetBufferSize_16u_C4R_Ctx(oSizeROI, nLevels, hpBufferSize, nppStreamCtx);
}

// ============================================================================
// nppiHistogramEvenGetBufferSize_16s_C4R - Get buffer size for 16s C4R
// ============================================================================

NppStatus nppiHistogramEvenGetBufferSize_16s_C4R_Ctx(NppiSize oSizeROI,
                                                     int nLevels[4],
                                                     int *hpBufferSize,
                                                     NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx; // Stream context not used for buffer size calculation

  if (!hpBufferSize || !nLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  return nppiHistogramEvenGetBufferSize_16s_C4R_Ctx_impl(oSizeROI, nLevels, hpBufferSize);
}

NppStatus nppiHistogramEvenGetBufferSize_16s_C4R(NppiSize oSizeROI,
                                                 int nLevels[4],
                                                 int *hpBufferSize) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiHistogramEvenGetBufferSize_16s_C4R_Ctx(oSizeROI, nLevels, hpBufferSize, nppStreamCtx);
}

// ============================================================================
// nppiHistogramEven_16u_C4R - 16-bit unsigned 4-channel histogram
// ============================================================================

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

NppStatus nppiHistogramEven_16u_C4R(const Npp16u *pSrc, int nSrcStep,
                                    NppiSize oSizeROI,
                                    Npp32s *pHist[4],
                                    int nLevels[4],
                                    Npp32s nLowerLevel[4],
                                    Npp32s nUpperLevel[4],
                                    Npp8u *pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiHistogramEven_16u_C4R_Ctx(pSrc, nSrcStep, oSizeROI, pHist,
                                       nLevels, nLowerLevel, nUpperLevel,
                                       pDeviceBuffer, nppStreamCtx);
}

// ============================================================================
// nppiHistogramEven_16s_C4R - 16-bit signed 4-channel histogram
// ============================================================================

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

NppStatus nppiHistogramEven_16s_C4R(const Npp16s *pSrc, int nSrcStep,
                                    NppiSize oSizeROI,
                                    Npp32s *pHist[4],
                                    int nLevels[4],
                                    Npp32s nLowerLevel[4],
                                    Npp32s nUpperLevel[4],
                                    Npp8u *pDeviceBuffer) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiHistogramEven_16s_C4R_Ctx(pSrc, nSrcStep, oSizeROI, pHist,
                                       nLevels, nLowerLevel, nUpperLevel,
                                       pDeviceBuffer, nppStreamCtx);
}
