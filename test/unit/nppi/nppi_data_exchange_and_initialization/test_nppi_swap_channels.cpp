#include "npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <iomanip>

using namespace npp_functional_test;

class SwapChannelsFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }

  // Helper function: Generate test image with identifiable channel values
  void generateTestImage_C4(std::vector<Npp8u> &data, int width, int height) {
    data.resize(width * height * 4);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * 4;
        // Channel 0: Red (value based on x position)
        data[idx + 0] = static_cast<Npp8u>((x * 255) / std::max(1, width - 1));
        // Channel 1: Green (value based on y position)
        data[idx + 1] = static_cast<Npp8u>((y * 255) / std::max(1, height - 1));
        // Channel 2: Blue (constant value)
        data[idx + 2] = 128;
        // Channel 3: Alpha (constant value)
        data[idx + 3] = 255;
      }
    }
  }

  // Helper function: Apply channel swap to expected data
  void applyChannelSwap(const std::vector<Npp8u> &src, std::vector<Npp8u> &dst,
                        int width, int height, const int aDstOrder[4]) {
    dst.resize(width * height * 4);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * 4;
        for (int c = 0; c < 4; c++) {
          dst[idx + c] = src[idx + aDstOrder[c]];
        }
      }
    }
  }

  // Helper function: Print image for debugging
  void printImage(const std::vector<Npp8u> &data, int width, int height,
                  const std::string &title) {
    std::cout << title << std::endl;
    for (int y = 0; y < std::min(4, height); y++) {
      for (int x = 0; x < std::min(4, width); x++) {
        int idx = (y * width + x) * 4;
        std::cout << "[" << std::setw(3) << (int)data[idx + 0] << ","
                  << std::setw(3) << (int)data[idx + 1] << ","
                  << std::setw(3) << (int)data[idx + 2] << ","
                  << std::setw(3) << (int)data[idx + 3] << "] ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
};

// ==================== nppiSwapChannels_8u_C4R Tests ====================

TEST_F(SwapChannelsFunctionalTest, SwapChannels_8u_C4R_RGBA_to_BGRA) {
  const int width = 8;
  const int height = 8;

  // Generate test data: RGBA format
  std::vector<Npp8u> src_data;
  generateTestImage_C4(src_data, width, height);

  // Expected result: BGRA (swap R and B channels)
  // aDstOrder = [2, 1, 0, 3] means:
  // dst[0] = src[2] (B), dst[1] = src[1] (G), dst[2] = src[0] (R), dst[3] = src[3] (A)
  int aDstOrder[4] = {2, 1, 0, 3};
  std::vector<Npp8u> expected_data;
  applyChannelSwap(src_data, expected_data, width, height, aDstOrder);

  // Allocate GPU memory
  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);

  src.copyFromHost(src_data);

  // Execute swap channels operation
  NppiSize roi = {width, height};
  NppStatus status = nppiSwapChannels_8u_C4R(src.get(), src.step(),
                                              dst.get(), dst.step(),
                                              roi, aDstOrder);

  ASSERT_EQ(status, NPP_SUCCESS) << "SwapChannels_8u_C4R failed";

  // Validate result
  std::vector<Npp8u> result;
  dst.copyToHost(result);

  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data))
      << "SwapChannels result incorrect for RGBA to BGRA";

  // Print for debugging
  printImage(src_data, width, height, "Original RGBA:");
  printImage(result, width, height, "Result BGRA:");
}

TEST_F(SwapChannelsFunctionalTest, SwapChannels_8u_C4R_ARGB_to_BGRA) {
  const int width = 4;
  const int height = 4;

  // Generate test data
  std::vector<Npp8u> src_data;
  generateTestImage_C4(src_data, width, height);

  // ARGB to BGRA: aDstOrder = [3, 2, 1, 0]
  int aDstOrder[4] = {3, 2, 1, 0};
  std::vector<Npp8u> expected_data;
  applyChannelSwap(src_data, expected_data, width, height, aDstOrder);

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);

  src.copyFromHost(src_data);

  NppiSize roi = {width, height};
  NppStatus status = nppiSwapChannels_8u_C4R(src.get(), src.step(),
                                              dst.get(), dst.step(),
                                              roi, aDstOrder);

  ASSERT_EQ(status, NPP_SUCCESS) << "SwapChannels_8u_C4R failed";

  std::vector<Npp8u> result;
  dst.copyToHost(result);

  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data))
      << "SwapChannels result incorrect for ARGB to BGRA";
}

TEST_F(SwapChannelsFunctionalTest, SwapChannels_8u_C4R_Ctx_RGBA_to_BGRA) {
  const int width = 8;
  const int height = 8;

  std::vector<Npp8u> src_data;
  generateTestImage_C4(src_data, width, height);

  int aDstOrder[4] = {2, 1, 0, 3};
  std::vector<Npp8u> expected_data;
  applyChannelSwap(src_data, expected_data, width, height, aDstOrder);

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);

  src.copyFromHost(src_data);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);

  NppStatus status = nppiSwapChannels_8u_C4R_Ctx(src.get(), src.step(),
                                                  dst.get(), dst.step(),
                                                  roi, aDstOrder, ctx);

  ASSERT_EQ(status, NPP_SUCCESS) << "SwapChannels_8u_C4R_Ctx failed";

  cudaStreamSynchronize(ctx.hStream);

  std::vector<Npp8u> result;
  dst.copyToHost(result);

  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data))
      << "SwapChannels_Ctx result incorrect";
}

// ==================== nppiSwapChannels_8u_C4IR Tests ====================

TEST_F(SwapChannelsFunctionalTest, SwapChannels_8u_C4IR_RGBA_to_BGRA) {
  const int width = 8;
  const int height = 8;

  // Generate test data
  std::vector<Npp8u> src_data;
  generateTestImage_C4(src_data, width, height);

  // Expected result after in-place swap
  int aDstOrder[4] = {2, 1, 0, 3};
  std::vector<Npp8u> expected_data;
  applyChannelSwap(src_data, expected_data, width, height, aDstOrder);

  // Allocate GPU memory (in-place operation)
  NppImageMemory<Npp8u> srcDst(width, height, 4);

  srcDst.copyFromHost(src_data);

  // Execute in-place swap channels operation
  NppiSize roi = {width, height};
  NppStatus status = nppiSwapChannels_8u_C4IR(srcDst.get(), srcDst.step(),
                                               roi, aDstOrder);

  ASSERT_EQ(status, NPP_SUCCESS) << "SwapChannels_8u_C4IR failed";

  // Validate result
  std::vector<Npp8u> result;
  srcDst.copyToHost(result);

  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data))
      << "SwapChannels in-place result incorrect";

  printImage(src_data, width, height, "Original RGBA:");
  printImage(result, width, height, "In-place Result BGRA:");
}

TEST_F(SwapChannelsFunctionalTest, SwapChannels_8u_C4IR_Identity) {
  const int width = 4;
  const int height = 4;

  std::vector<Npp8u> src_data;
  generateTestImage_C4(src_data, width, height);

  // Identity swap: no change
  int aDstOrder[4] = {0, 1, 2, 3};
  std::vector<Npp8u> expected_data = src_data;

  NppImageMemory<Npp8u> srcDst(width, height, 4);
  srcDst.copyFromHost(src_data);

  NppiSize roi = {width, height};
  NppStatus status = nppiSwapChannels_8u_C4IR(srcDst.get(), srcDst.step(),
                                               roi, aDstOrder);

  ASSERT_EQ(status, NPP_SUCCESS) << "SwapChannels_8u_C4IR identity failed";

  std::vector<Npp8u> result;
  srcDst.copyToHost(result);

  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data))
      << "Identity swap should not change data";
}

TEST_F(SwapChannelsFunctionalTest, SwapChannels_8u_C4IR_Ctx_RGBA_to_BGRA) {
  const int width = 8;
  const int height = 8;

  std::vector<Npp8u> src_data;
  generateTestImage_C4(src_data, width, height);

  int aDstOrder[4] = {2, 1, 0, 3};
  std::vector<Npp8u> expected_data;
  applyChannelSwap(src_data, expected_data, width, height, aDstOrder);

  NppImageMemory<Npp8u> srcDst(width, height, 4);
  srcDst.copyFromHost(src_data);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);

  NppStatus status = nppiSwapChannels_8u_C4IR_Ctx(srcDst.get(), srcDst.step(),
                                                   roi, aDstOrder, ctx);

  ASSERT_EQ(status, NPP_SUCCESS) << "SwapChannels_8u_C4IR_Ctx failed";

  cudaStreamSynchronize(ctx.hStream);

  std::vector<Npp8u> result;
  srcDst.copyToHost(result);

  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data))
      << "SwapChannels in-place Ctx result incorrect";
}

// ==================== Edge Cases ====================

TEST_F(SwapChannelsFunctionalTest, SwapChannels_8u_C4R_LargeImage) {
  const int width = 1920;
  const int height = 1080;

  std::vector<Npp8u> src_data;
  generateTestImage_C4(src_data, width, height);

  int aDstOrder[4] = {2, 1, 0, 3};
  std::vector<Npp8u> expected_data;
  applyChannelSwap(src_data, expected_data, width, height, aDstOrder);

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);

  src.copyFromHost(src_data);

  NppiSize roi = {width, height};
  NppStatus status = nppiSwapChannels_8u_C4R(src.get(), src.step(),
                                              dst.get(), dst.step(),
                                              roi, aDstOrder);

  ASSERT_EQ(status, NPP_SUCCESS) << "SwapChannels large image failed";

  std::vector<Npp8u> result;
  dst.copyToHost(result);

  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data))
      << "Large image swap failed";
}

TEST_F(SwapChannelsFunctionalTest, SwapChannels_8u_C4R_MinimalSize) {
  const int width = 1;
  const int height = 1;

  std::vector<Npp8u> src_data = {10, 20, 30, 40};
  int aDstOrder[4] = {2, 1, 0, 3};
  std::vector<Npp8u> expected_data = {30, 20, 10, 40};

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);

  src.copyFromHost(src_data);

  NppiSize roi = {width, height};
  NppStatus status = nppiSwapChannels_8u_C4R(src.get(), src.step(),
                                              dst.get(), dst.step(),
                                              roi, aDstOrder);

  ASSERT_EQ(status, NPP_SUCCESS) << "SwapChannels 1x1 failed";

  std::vector<Npp8u> result;
  dst.copyToHost(result);

  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data))
      << "1x1 image swap failed";
}
