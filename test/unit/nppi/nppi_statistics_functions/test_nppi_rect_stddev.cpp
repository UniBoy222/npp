#include "npp.h"
#include "framework/npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

using namespace npp_functional_test;

/**
 * @brief NPP RectStdDev Tests
 *
 * Tests for nppiRectStdDev_32s32f_C1R family of functions.
 * These tests verify rectangular standard deviation operations.
 */
class NPPIRectStdDevTest : public NppTestBase {
protected:
  void SetUp() override {
    NppTestBase::SetUp();
    width = 16;
    height = 16;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;
};

/**
 * @brief Test nppiRectStdDev_32s32f_C1R basic functionality
 *
 * Verifies rectangular standard deviation computation with a simple test case.
 */
TEST_F(NPPIRectStdDevTest, RectStdDev_32s32f_C1R) {
  size_t dataSize = width * height;

  // Create source data with known pattern
  std::vector<Npp32s> srcData(dataSize);
  std::vector<Npp64f> sqrData(dataSize);

  for (size_t i = 0; i < dataSize; i++) {
    srcData[i] = static_cast<Npp32s>(i % 10);
    sqrData[i] = static_cast<Npp64f>(srcData[i] * srcData[i]);
  }

  NppImageMemory<Npp32s> src(width, height, 1);
  NppImageMemory<Npp64f> sqr(width, height, 1);
  NppImageMemory<Npp32f> dst(width, height, 1);

  src.copyFromHost(srcData);
  sqr.copyFromHost(sqrData);

  // Define a 3x3 rectangular window
  NppiRect rect = {1, 1, 1, 1}; // x, y, width, height

  NppStatus status = nppiRectStdDev_32s32f_C1R(
      src.get(), src.step(),
      sqr.get(), sqr.step(),
      dst.get(), dst.step(),
      roi, rect
  );

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> dstData;
  dst.copyToHost(dstData);

  // Verify that output contains valid standard deviation values
  for (size_t i = 0; i < dataSize; i++) {
    EXPECT_GE(dstData[i], 0.0f) << "Standard deviation should be non-negative at index " << i;
    EXPECT_LT(dstData[i], 100.0f) << "Standard deviation should be reasonable at index " << i;
  }
}

/**
 * @brief Test nppiRectStdDev_32s32f_C1R_Ctx with stream context
 *
 * Verifies the context-aware version that supports custom CUDA streams.
 */
TEST_F(NPPIRectStdDevTest, RectStdDev_32s32f_C1R_Ctx) {
  size_t dataSize = width * height;

  std::vector<Npp32s> srcData(dataSize);
  std::vector<Npp64f> sqrData(dataSize);

  for (size_t i = 0; i < dataSize; i++) {
    srcData[i] = static_cast<Npp32s>(i % 10);
    sqrData[i] = static_cast<Npp64f>(srcData[i] * srcData[i]);
  }

  NppImageMemory<Npp32s> src(width, height, 1);
  NppImageMemory<Npp64f> sqr(width, height, 1);
  NppImageMemory<Npp32f> dst(width, height, 1);

  src.copyFromHost(srcData);
  sqr.copyFromHost(sqrData);

  NppiRect rect = {1, 1, 1, 1};

  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status = nppiRectStdDev_32s32f_C1R_Ctx(
      src.get(), src.step(),
      sqr.get(), sqr.step(),
      dst.get(), dst.step(),
      roi, rect,
      nppStreamCtx
  );

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> dstData;
  dst.copyToHost(dstData);

  // Verify output validity
  for (size_t i = 0; i < dataSize; i++) {
    EXPECT_GE(dstData[i], 0.0f);
  }
}
