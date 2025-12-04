#include "npp.h"
#include "framework/npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>

using namespace npp_functional_test;

/**
 * @brief NPP HistogramEven Tests
 *
 * Tests for nppiHistogramEven_16u_C4R and nppiHistogramEven_16s_C4R families.
 * These tests verify basic histogram computation on 4-channel 16-bit images.
 */
class NPPIHistogramEvenC4RTest : public NppTestBase {
protected:
  void SetUp() override {
    NppTestBase::SetUp();
    width = 8;
    height = 8;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;
};

/**
 * @brief Test nppiHistogramEven_16u_C4R basic functionality
 *
 * Verifies histogram computation for 16-bit unsigned 4-channel images.
 */
TEST_F(NPPIHistogramEvenC4RTest, HistogramEven_16u_C4R) {
  size_t dataSize = width * height * 4;

  // Generate test data with known distribution
  std::vector<Npp16u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 4) {
    srcData[i + 0] = 100;  // Channel 0
    srcData[i + 1] = 200;  // Channel 1
    srcData[i + 2] = 300;  // Channel 2
    srcData[i + 3] = 400;  // Channel 3
  }

  NppImageMemory<Npp16u> src(width, height, 4);
  src.copyFromHost(srcData);

  // Setup histogram parameters
  int nLevels[4] = {256, 256, 256, 256};
  Npp32s nLowerLevel[4] = {0, 0, 0, 0};
  Npp32s nUpperLevel[4] = {512, 512, 512, 512};

  // Allocate histogram buffers
  DeviceMemory<Npp32s> hist0{nLevels[0] - 1};
  DeviceMemory<Npp32s> hist1{nLevels[1] - 1};
  DeviceMemory<Npp32s> hist2{nLevels[2] - 1};
  DeviceMemory<Npp32s> hist3{nLevels[3] - 1};

  Npp32s *pHist[4] = {hist0.get(), hist1.get(), hist2.get(), hist3.get()};

  // Get buffer size
  size_t bufferSize = 0;
  NppStatus status = nppiHistogramEvenGetBufferSize_16u_C4R(roi, nLevels, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);

  DeviceMemory<Npp8u> deviceBuffer(bufferSize);

  // Compute histogram
  status = nppiHistogramEven_16u_C4R(
      src.get(), src.step(),
      roi, pHist, nLevels, nLowerLevel, nUpperLevel,
      deviceBuffer.get()
  );

  EXPECT_EQ(status, NPP_SUCCESS);

  // Verify histogram results
  std::vector<Npp32s> histData0, histData1;
  hist0.copyToHost(histData0);
  hist1.copyToHost(histData1);

  // All pixels in channel 0 have value 100, which maps to bin 49 (100 * 255 / 512)
  int expectedBin0 = (100 * (nLevels[0] - 1)) / (nUpperLevel[0] - nLowerLevel[0]);
  EXPECT_GT(histData0[expectedBin0], 0);

  // All pixels in channel 1 have value 200, which maps to bin 99
  int expectedBin1 = (200 * (nLevels[1] - 1)) / (nUpperLevel[1] - nLowerLevel[1]);
  EXPECT_GT(histData1[expectedBin1], 0);
}

/**
 * @brief Test nppiHistogramEven_16u_C4R_Ctx with stream context
 *
 * Verifies the context-aware version that supports custom CUDA streams.
 */
TEST_F(NPPIHistogramEvenC4RTest, HistogramEven_16u_C4R_Ctx) {
  size_t dataSize = width * height * 4;

  std::vector<Npp16u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 4) {
    srcData[i + 0] = 100;
    srcData[i + 1] = 200;
    srcData[i + 2] = 300;
    srcData[i + 3] = 400;
  }

  NppImageMemory<Npp16u> src(width, height, 4);
  src.copyFromHost(srcData);

  int nLevels[4] = {256, 256, 256, 256};
  Npp32s nLowerLevel[4] = {0, 0, 0, 0};
  Npp32s nUpperLevel[4] = {512, 512, 512, 512};

  DeviceMemory<Npp32s> hist0{nLevels[0] - 1};
  DeviceMemory<Npp32s> hist1{nLevels[1] - 1};
  DeviceMemory<Npp32s> hist2{nLevels[2] - 1};
  DeviceMemory<Npp32s> hist3{nLevels[3] - 1};

  Npp32s *pHist[4] = {hist0.get(), hist1.get(), hist2.get(), hist3.get()};

  size_t bufferSize = 0;
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status = nppiHistogramEvenGetBufferSize_16u_C4R_Ctx(roi, nLevels, &bufferSize, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  DeviceMemory<Npp8u> deviceBuffer(bufferSize);

  status = nppiHistogramEven_16u_C4R_Ctx(
      src.get(), src.step(),
      roi, pHist, nLevels, nLowerLevel, nUpperLevel,
      deviceBuffer.get(), nppStreamCtx
  );

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32s> histData0;
  hist0.copyToHost(histData0);

  int expectedBin0 = (100 * (nLevels[0] - 1)) / (nUpperLevel[0] - nLowerLevel[0]);
  EXPECT_GT(histData0[expectedBin0], 0);
}

/**
 * @brief Test nppiHistogramEven_16s_C4R basic functionality
 *
 * Verifies histogram computation for 16-bit signed 4-channel images.
 */
TEST_F(NPPIHistogramEvenC4RTest, HistogramEven_16s_C4R) {
  size_t dataSize = width * height * 4;

  // Generate test data with signed values
  std::vector<Npp16s> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 4) {
    srcData[i + 0] = -100;  // Channel 0
    srcData[i + 1] = 0;     // Channel 1
    srcData[i + 2] = 100;   // Channel 2
    srcData[i + 3] = 200;   // Channel 3
  }

  NppImageMemory<Npp16s> src(width, height, 4);
  src.copyFromHost(srcData);

  // Setup histogram parameters for signed data
  int nLevels[4] = {256, 256, 256, 256};
  Npp32s nLowerLevel[4] = {-256, -256, -256, -256};
  Npp32s nUpperLevel[4] = {256, 256, 256, 256};

  DeviceMemory<Npp32s> hist0{nLevels[0] - 1};
  DeviceMemory<Npp32s> hist1{nLevels[1] - 1};
  DeviceMemory<Npp32s> hist2{nLevels[2] - 1};
  DeviceMemory<Npp32s> hist3{nLevels[3] - 1};

  Npp32s *pHist[4] = {hist0.get(), hist1.get(), hist2.get(), hist3.get()};

  size_t bufferSize = 0;
  NppStatus status = nppiHistogramEvenGetBufferSize_16s_C4R(roi, nLevels, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);

  DeviceMemory<Npp8u> deviceBuffer(bufferSize);

  status = nppiHistogramEven_16s_C4R(
      src.get(), src.step(),
      roi, pHist, nLevels, nLowerLevel, nUpperLevel,
      deviceBuffer.get()
  );

  EXPECT_EQ(status, NPP_SUCCESS);

  // Verify histogram results
  std::vector<Npp32s> histData0, histData1, histData2;
  hist0.copyToHost(histData0);
  hist1.copyToHost(histData1);
  hist2.copyToHost(histData2);

  // Channel 0: value -100 maps to bin ((-100 - (-256)) * 255 / 512) = 76
  int expectedBin0 = ((-100 - nLowerLevel[0]) * (nLevels[0] - 1)) / (nUpperLevel[0] - nLowerLevel[0]);
  EXPECT_GT(histData0[expectedBin0], 0);

  // Channel 1: value 0 maps to bin ((0 - (-256)) * 255 / 512) = 127
  int expectedBin1 = ((0 - nLowerLevel[1]) * (nLevels[1] - 1)) / (nUpperLevel[1] - nLowerLevel[1]);
  EXPECT_GT(histData1[expectedBin1], 0);

  // Channel 2: value 100 maps to bin ((100 - (-256)) * 255 / 512) = 177
  int expectedBin2 = ((100 - nLowerLevel[2]) * (nLevels[2] - 1)) / (nUpperLevel[2] - nLowerLevel[2]);
  EXPECT_GT(histData2[expectedBin2], 0);
}

/**
 * @brief Test nppiHistogramEven_16s_C4R_Ctx with stream context
 *
 * Verifies the context-aware version for signed 16-bit data.
 */
TEST_F(NPPIHistogramEvenC4RTest, HistogramEven_16s_C4R_Ctx) {
  size_t dataSize = width * height * 4;

  std::vector<Npp16s> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 4) {
    srcData[i + 0] = -100;
    srcData[i + 1] = 0;
    srcData[i + 2] = 100;
    srcData[i + 3] = 200;
  }

  NppImageMemory<Npp16s> src(width, height, 4);
  src.copyFromHost(srcData);

  int nLevels[4] = {256, 256, 256, 256};
  Npp32s nLowerLevel[4] = {-256, -256, -256, -256};
  Npp32s nUpperLevel[4] = {256, 256, 256, 256};

  DeviceMemory<Npp32s> hist0{nLevels[0] - 1};
  DeviceMemory<Npp32s> hist1{nLevels[1] - 1};
  DeviceMemory<Npp32s> hist2{nLevels[2] - 1};
  DeviceMemory<Npp32s> hist3{nLevels[3] - 1};

  Npp32s *pHist[4] = {hist0.get(), hist1.get(), hist2.get(), hist3.get()};

  size_t bufferSize = 0;
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status = nppiHistogramEvenGetBufferSize_16s_C4R_Ctx(roi, nLevels, &bufferSize, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  DeviceMemory<Npp8u> deviceBuffer(bufferSize);

  status = nppiHistogramEven_16s_C4R_Ctx(
      src.get(), src.step(),
      roi, pHist, nLevels, nLowerLevel, nUpperLevel,
      deviceBuffer.get(), nppStreamCtx
  );

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32s> histData1;
  hist1.copyToHost(histData1);

  int expectedBin1 = ((0 - nLowerLevel[1]) * (nLevels[1] - 1)) / (nUpperLevel[1] - nLowerLevel[1]);
  EXPECT_GT(histData1[expectedBin1], 0);
}
