#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for implementation functions
extern "C" {
NppStatus nppiCopyConstBorder_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep,
                                               NppiSize oSrcSizeROI,
                                               Npp8u *pDst, int nDstStep,
                                               NppiSize oDstSizeROI,
                                               int nTopBorderHeight, int nLeftBorderWidth,
                                               Npp8u nValue,
                                               NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateCopyConstBorderInputs(const void *pSrc, int nSrcStep,
                                                       NppiSize oSrcSizeROI,
                                                       void *pDst, int nDstStep,
                                                       NppiSize oDstSizeROI,
                                                       int nTopBorderHeight, int nLeftBorderWidth) {
  if (oSrcSizeROI.width <= 0 || oSrcSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (oDstSizeROI.width <= 0 || oDstSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (nTopBorderHeight < 0 || nLeftBorderWidth < 0) {
    return NPP_BORDER_ERROR;
  }

  // Validate that destination size is large enough to contain source + borders
  // Note: The API allows flexible border sizes, so we just check basic constraints
  if (oDstSizeROI.width < oSrcSizeROI.width || oDstSizeROI.height < oSrcSizeROI.height) {
    return NPP_SIZE_ERROR;
  }

  return NPP_SUCCESS;
}

// ============================================================================
// nppiCopyConstBorder_8u_C1R - Copy with constant border
// ============================================================================

NppStatus nppiCopyConstBorder_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI,
                                          Npp8u *pDst, int nDstStep,
                                          NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          Npp8u nValue,
                                          NppStreamContext nppStreamCtx) {
  NppStatus status = validateCopyConstBorderInputs(pSrc, nSrcStep, oSrcSizeROI,
                                                     pDst, nDstStep, oDstSizeROI,
                                                     nTopBorderHeight, nLeftBorderWidth);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiCopyConstBorder_8u_C1R_Ctx_impl(pSrc, nSrcStep, oSrcSizeROI,
                                              pDst, nDstStep, oDstSizeROI,
                                              nTopBorderHeight, nLeftBorderWidth,
                                              nValue, nppStreamCtx);
}

NppStatus nppiCopyConstBorder_8u_C1R(const Npp8u *pSrc, int nSrcStep,
                                      NppiSize oSrcSizeROI,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      Npp8u nValue) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiCopyConstBorder_8u_C1R_Ctx(pSrc, nSrcStep, oSrcSizeROI,
                                         pDst, nDstStep, oDstSizeROI,
                                         nTopBorderHeight, nLeftBorderWidth,
                                         nValue, nppStreamCtx);
}
