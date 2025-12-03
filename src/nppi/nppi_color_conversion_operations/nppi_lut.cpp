#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for mpp host func implementations

extern "C" {
// 8u implementations
NppStatus nppiLUT_Linear_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                         const Npp32s *pValues, const Npp32s *pLevels, int nLevels,
                                         NppStreamContext nppStreamCtx);
NppStatus nppiLUT_Linear_8u_C1IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                          const Npp32s *pValues, const Npp32s *pLevels, int nLevels,
                                          NppStreamContext nppStreamCtx);
NppStatus nppiLUT_Linear_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                         const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3],
                                         NppStreamContext nppStreamCtx);
NppStatus nppiLUT_Linear_8u_C3IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                          const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3],
                                          NppStreamContext nppStreamCtx);
NppStatus nppiLUT_Linear_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                         const Npp32s *pValues[4], const Npp32s *pLevels[4], int nLevels[4],
                                         NppStreamContext nppStreamCtx);
NppStatus nppiLUT_Linear_8u_C4IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                          const Npp32s *pValues[4], const Npp32s *pLevels[4], int nLevels[4],
                                          NppStreamContext nppStreamCtx);
NppStatus nppiLUT_Linear_8u_AC4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                          const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3],
                                          NppStreamContext nppStreamCtx);
NppStatus nppiLUT_Linear_8u_AC4IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                           const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3],
                                           NppStreamContext nppStreamCtx);

// 16u implementations
NppStatus nppiLUT_Linear_16u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                          const Npp32s *pValues, const Npp32s *pLevels, int nLevels,
                                          NppStreamContext nppStreamCtx);
NppStatus nppiLUT_Linear_16u_C3R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                          const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3],
                                          NppStreamContext nppStreamCtx);

// 16s implementations
NppStatus nppiLUT_Linear_16s_C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                          const Npp32s *pValues, const Npp32s *pLevels, int nLevels,
                                          NppStreamContext nppStreamCtx);
NppStatus nppiLUT_Linear_16s_C3R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                          const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3],
                                          NppStreamContext nppStreamCtx);

// 32f implementations
NppStatus nppiLUT_Linear_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                          const Npp32s *pValues, const Npp32s *pLevels, int nLevels,
                                          NppStreamContext nppStreamCtx);
NppStatus nppiLUT_Linear_32f_C3R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                          const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3],
                                          NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateLUTLinearInputs(const void *pSrc, int nSrcStep, void *pDst, int nDstStep,
                                                NppiSize oSizeROI, const void *pValues, const void *pLevels,
                                                int nLevels) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (!pValues || !pLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (nLevels < 2) { // At least 2 level points needed for linear interpolation
    return NPP_LUT_NUMBER_OF_LEVELS_ERROR;
  }

  return NPP_SUCCESS;
}

// ============================================================================
// 8-bit unsigned single channel linear LUT
// ============================================================================

NppStatus nppiLUT_Linear_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp32s *pValues, const Npp32s *pLevels, int nLevels,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status = validateLUTLinearInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiLUT_Linear_8u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels,
                                        nppStreamCtx);
}

NppStatus nppiLUT_Linear_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                const Npp32s *pValues, const Npp32s *pLevels, int nLevels) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiLUT_Linear_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

// ============================================================================
// 8-bit unsigned single channel in-place linear LUT
// ============================================================================

NppStatus nppiLUT_Linear_8u_C1IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                     const Npp32s *pValues, const Npp32s *pLevels, int nLevels,
                                     NppStreamContext nppStreamCtx) {
  if (!pSrcDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (!pValues || !pLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (nLevels < 2) {
    return NPP_LUT_NUMBER_OF_LEVELS_ERROR;
  }

  return nppiLUT_Linear_8u_C1IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Linear_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                 const Npp32s *pValues, const Npp32s *pLevels, int nLevels) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiLUT_Linear_8u_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

// ============================================================================
// 8-bit unsigned three channel linear LUT
// ============================================================================

NppStatus nppiLUT_Linear_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3],
                                    NppStreamContext nppStreamCtx) {
  // Validate basic parameters
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (!pValues || !pLevels || !nLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // Validate parameters for each channel
  for (int c = 0; c < 3; c++) {
    if (!pValues[c] || !pLevels[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
    if (nLevels[c] < 2) {
      return NPP_LUT_NUMBER_OF_LEVELS_ERROR;
    }
  }

  return nppiLUT_Linear_8u_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels,
                                        nppStreamCtx);
}

NppStatus nppiLUT_Linear_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiLUT_Linear_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

// ============================================================================
// 8-bit unsigned three channel in-place linear LUT
// ============================================================================

NppStatus nppiLUT_Linear_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                     const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3],
                                     NppStreamContext nppStreamCtx) {
  if (!pSrcDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (!pValues || !pLevels || !nLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // Validate parameters for each channel
  for (int c = 0; c < 3; c++) {
    if (!pValues[c] || !pLevels[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
    if (nLevels[c] < 2) {
      return NPP_LUT_NUMBER_OF_LEVELS_ERROR;
    }
  }

  return nppiLUT_Linear_8u_C3IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Linear_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                 const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiLUT_Linear_8u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

// ============================================================================
// 8-bit unsigned four channel linear LUT
// ============================================================================

NppStatus nppiLUT_Linear_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp32s *pValues[4], const Npp32s *pLevels[4], int nLevels[4],
                                    NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (!pValues || !pLevels || !nLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // Validate parameters for each channel
  for (int c = 0; c < 4; c++) {
    if (!pValues[c] || !pLevels[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
    if (nLevels[c] < 2) {
      return NPP_LUT_NUMBER_OF_LEVELS_ERROR;
    }
  }

  return nppiLUT_Linear_8u_C4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Linear_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                const Npp32s *pValues[4], const Npp32s *pLevels[4], int nLevels[4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiLUT_Linear_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

// ============================================================================
// 8-bit unsigned four channel in-place linear LUT
// ============================================================================

NppStatus nppiLUT_Linear_8u_C4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                     const Npp32s *pValues[4], const Npp32s *pLevels[4], int nLevels[4],
                                     NppStreamContext nppStreamCtx) {
  if (!pSrcDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (!pValues || !pLevels || !nLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // Validate parameters for each channel
  for (int c = 0; c < 4; c++) {
    if (!pValues[c] || !pLevels[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
    if (nLevels[c] < 2) {
      return NPP_LUT_NUMBER_OF_LEVELS_ERROR;
    }
  }

  return nppiLUT_Linear_8u_C4IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Linear_8u_C4IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                 const Npp32s *pValues[4], const Npp32s *pLevels[4], int nLevels[4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiLUT_Linear_8u_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

// ============================================================================
// 8-bit unsigned AC4 linear LUT (4-channel, not affecting Alpha)
// ============================================================================

NppStatus nppiLUT_Linear_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3],
                                     NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (!pValues || !pLevels || !nLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // Validate parameters for each channel (only 3 channels for AC4)
  for (int c = 0; c < 3; c++) {
    if (!pValues[c] || !pLevels[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
    if (nLevels[c] < 2) {
      return NPP_LUT_NUMBER_OF_LEVELS_ERROR;
    }
  }

  return nppiLUT_Linear_8u_AC4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Linear_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                 const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiLUT_Linear_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

// ============================================================================
// 8-bit unsigned AC4 in-place linear LUT
// ============================================================================

NppStatus nppiLUT_Linear_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                      const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3],
                                      NppStreamContext nppStreamCtx) {
  if (!pSrcDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (!pValues || !pLevels || !nLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // Validate parameters for each channel (only 3 channels for AC4)
  for (int c = 0; c < 3; c++) {
    if (!pValues[c] || !pLevels[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
    if (nLevels[c] < 2) {
      return NPP_LUT_NUMBER_OF_LEVELS_ERROR;
    }
  }

  return nppiLUT_Linear_8u_AC4IR_Ctx_impl(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Linear_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiLUT_Linear_8u_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

// ============================================================================
// 16-bit unsigned linear LUT
// ============================================================================

NppStatus nppiLUT_Linear_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp32s *pValues, const Npp32s *pLevels, int nLevels,
                                     NppStreamContext nppStreamCtx) {
  NppStatus status = validateLUTLinearInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiLUT_Linear_16u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Linear_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 const Npp32s *pValues, const Npp32s *pLevels, int nLevels) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiLUT_Linear_16u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Linear_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3],
                                     NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (!pValues || !pLevels || !nLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  for (int c = 0; c < 3; c++) {
    if (!pValues[c] || !pLevels[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
    if (nLevels[c] < 2) {
      return NPP_LUT_NUMBER_OF_LEVELS_ERROR;
    }
  }

  return nppiLUT_Linear_16u_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Linear_16u_C3R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiLUT_Linear_16u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

// ============================================================================
// 16-bit signed linear LUT
// ============================================================================

NppStatus nppiLUT_Linear_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp32s *pValues, const Npp32s *pLevels, int nLevels,
                                     NppStreamContext nppStreamCtx) {
  NppStatus status = validateLUTLinearInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiLUT_Linear_16s_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Linear_16s_C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 const Npp32s *pValues, const Npp32s *pLevels, int nLevels) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiLUT_Linear_16s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Linear_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3],
                                     NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (!pValues || !pLevels || !nLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  for (int c = 0; c < 3; c++) {
    if (!pValues[c] || !pLevels[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
    if (nLevels[c] < 2) {
      return NPP_LUT_NUMBER_OF_LEVELS_ERROR;
    }
  }

  return nppiLUT_Linear_16s_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Linear_16s_C3R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiLUT_Linear_16s_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

// ============================================================================
// 32-bit float linear LUT
// ============================================================================

NppStatus nppiLUT_Linear_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp32s *pValues, const Npp32s *pLevels, int nLevels,
                                     NppStreamContext nppStreamCtx) {
  NppStatus status = validateLUTLinearInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiLUT_Linear_32f_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Linear_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                 const Npp32s *pValues, const Npp32s *pLevels, int nLevels) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiLUT_Linear_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Linear_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3],
                                     NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (!pValues || !pLevels || !nLevels) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  for (int c = 0; c < 3; c++) {
    if (!pValues[c] || !pLevels[c]) {
      return NPP_NULL_POINTER_ERROR;
    }
    if (nLevels[c] < 2) {
      return NPP_LUT_NUMBER_OF_LEVELS_ERROR;
    }
  }

  return nppiLUT_Linear_32f_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}

NppStatus nppiLUT_Linear_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                 const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiLUT_Linear_32f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, nLevels, nppStreamCtx);
}
