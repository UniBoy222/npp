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

  void TearDown() override {
    // Clean up device memory
    if (d_src) cudaFree(d_src);
    if (d_sqr) cudaFree(d_sqr);
    if (d_dst) cudaFree(d_dst);
    d_src = nullptr;
    d_sqr = nullptr;
    d_dst = nullptr;

    NppTestBase::TearDown();
  }

  int width, height;
  NppiSize roi;
  Npp32s *d_src = nullptr;
  Npp64f *d_sqr = nullptr;
  Npp32f *d_dst = nullptr;
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

  // Allocate device memory manually
  int srcStep = width * sizeof(Npp32s);
  int sqrStep = width * sizeof(Npp64f);
  int dstStep = width * sizeof(Npp32f);

  ASSERT_EQ(cudaMalloc(&d_src, dataSize * sizeof(Npp32s)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_sqr, dataSize * sizeof(Npp64f)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_dst, dataSize * sizeof(Npp32f)), cudaSuccess);

  ASSERT_EQ(cudaMemcpy(d_src, srcData.data(), dataSize * sizeof(Npp32s), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_sqr, sqrData.data(), dataSize * sizeof(Npp64f), cudaMemcpyHostToDevice), cudaSuccess);

  // Define a 3x3 rectangular window
  NppiRect rect = {1, 1, 1, 1}; // x, y, width, height

  NppStatus status = nppiRectStdDev_32s32f_C1R(
      d_src, srcStep,
      d_sqr, sqrStep,
      d_dst, dstStep,
      roi, rect
  );

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> dstData(dataSize);
  ASSERT_EQ(cudaMemcpy(dstData.data(), d_dst, dataSize * sizeof(Npp32f), cudaMemcpyDeviceToHost), cudaSuccess);

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

  // Allocate device memory manually
  int srcStep = width * sizeof(Npp32s);
  int sqrStep = width * sizeof(Npp64f);
  int dstStep = width * sizeof(Npp32f);

  ASSERT_EQ(cudaMalloc(&d_src, dataSize * sizeof(Npp32s)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_sqr, dataSize * sizeof(Npp64f)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_dst, dataSize * sizeof(Npp32f)), cudaSuccess);

  ASSERT_EQ(cudaMemcpy(d_src, srcData.data(), dataSize * sizeof(Npp32s), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_sqr, sqrData.data(), dataSize * sizeof(Npp64f), cudaMemcpyHostToDevice), cudaSuccess);

  NppiRect rect = {1, 1, 1, 1};

  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status = nppiRectStdDev_32s32f_C1R_Ctx(
      d_src, srcStep,
      d_sqr, sqrStep,
      d_dst, dstStep,
      roi, rect,
      nppStreamCtx
  );

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> dstData(dataSize);
  ASSERT_EQ(cudaMemcpy(dstData.data(), d_dst, dataSize * sizeof(Npp32f), cudaMemcpyDeviceToHost), cudaSuccess);

  // Verify output validity
  for (size_t i = 0; i < dataSize; i++) {
    EXPECT_GE(dstData[i], 0.0f);
  }
}
