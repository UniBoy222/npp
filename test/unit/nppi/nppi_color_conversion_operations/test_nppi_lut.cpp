#include "npp.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

class NPPILUTTest : public ::testing::Test {
protected:
  void SetUp() override {
    width = 16;
    height = 12;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;
};

// 测试8位无符号单通道线性LUT
TEST_F(NPPILUTTest, LUT_Linear_8u_C1R_Basic) {
  size_t dataSize = width * height;
  std::vector<Npp8u> srcData(dataSize), dstData(dataSize);

  // 生成测试数据：0-255的渐变
  for (size_t i = 0; i < dataSize; i++) {
    srcData[i] = (Npp8u)(i % 256);
  }

  // 创建LUT：反转映射 (255-x)
  int nLevels = 3;
  std::vector<Npp32s> pLevels = {0, 128, 255};
  std::vector<Npp32s> pValues = {255, 127, 0};

  // 分配GPU内存使用NPP函数
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);

  // 使用RAII模式确保内存清理
  struct ResourceGuard {
    Npp8u *src, *dst;
    ResourceGuard(Npp8u *s, Npp8u *d) : src(s), dst(d) {}
    ~ResourceGuard() {
      if (src)
        nppiFree(src);
      if (dst)
        nppiFree(dst);
    }
  } guard(d_src, d_dst);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 按行复制数据到GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width, width * sizeof(Npp8u), cudaMemcpyHostToDevice);
  }

  // NPP函数期望host指针，不需要手动分配device内存
  // 函数内部会自动处理device内存的分配和拷贝
  NppStatus status = nppiLUT_Linear_8u_C1R(d_src, srcStep, d_dst, dstStep, roi, pValues.data(), pLevels.data(), nLevels);
  std::cout << "NPP status: " << status << std::endl;

  EXPECT_EQ(status, NPP_SUCCESS);

  // 按行拷贝结果回主机
  for (int y = 0; y < height; y++) {
    cudaMemcpy(dstData.data() + y * width, (char *)d_dst + y * dstStep, width * sizeof(Npp8u), cudaMemcpyDeviceToHost);
  }

  // Validate结果：检查几个关键点
  EXPECT_EQ(dstData[0], 255);   // 输入0应该映射到255
  EXPECT_EQ(dstData[128], 127); // 输入128应该映射到127
  if (dataSize > 255) {
    // 只有当数据足够大时才测试255索引
    size_t idx255 = 255;
    EXPECT_EQ(dstData[idx255], 0); // 输入255应该映射到0
  }

  // 资源将由ResourceGuard自动清理
}

// 测试8位无符号三通道线性LUT
TEST_F(NPPILUTTest, LUT_Linear_8u_C3R_Basic) {
  const int channels = 3;
  size_t dataSize = width * height * channels;
  std::vector<Npp8u> srcData(dataSize), dstData(dataSize);

  // 生成测试数据
  for (size_t i = 0; i < dataSize; i++) {
    srcData[i] = (Npp8u)(i % 256);
  }

  // 创建LUT：每个通道不同的映射
  int nLevels[3] = {3, 3, 3};
  std::vector<Npp32s> pLevels0 = {0, 128, 255};
  std::vector<Npp32s> pValues0 = {255, 127, 0};
  std::vector<Npp32s> pLevels1 = {0, 128, 255};
  std::vector<Npp32s> pValues1 = {0, 127, 255};
  std::vector<Npp32s> pLevels2 = {0, 128, 255};
  std::vector<Npp32s> pValues2 = {128, 128, 128};

  // 分配GPU内存
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C3(width, height, &dstStep);

  struct ResourceGuard {
    Npp8u *src, *dst;
    ResourceGuard(Npp8u *s, Npp8u *d) : src(s), dst(d) {}
    ~ResourceGuard() {
      if (src) nppiFree(src);
      if (dst) nppiFree(dst);
    }
  } guard(d_src, d_dst);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据到GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * channels,
               width * channels * sizeof(Npp8u), cudaMemcpyHostToDevice);
  }

  // NPP函数期望host指针数组
  const Npp32s *pValues[3] = {pValues0.data(), pValues1.data(), pValues2.data()};
  const Npp32s *pLevels[3] = {pLevels0.data(), pLevels1.data(), pLevels2.data()};

  // 执行LUT
  NppStatus status = nppiLUT_Linear_8u_C3R(d_src, srcStep, d_dst, dstStep, roi, pValues, pLevels, nLevels);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  for (int y = 0; y < height; y++) {
    cudaMemcpy(dstData.data() + y * width * channels, (char *)d_dst + y * dstStep,
               width * channels * sizeof(Npp8u), cudaMemcpyDeviceToHost);
  }

  // 验证结果：检查几个关键点
  // 通道0: 反转映射 (255->0, 128->127, 0->255)
  // 通道1: 正向映射 (0->0, 128->127, 255->255)
  // 通道2: 恒定映射 (所有值->128)
  EXPECT_EQ(dstData[0], 255);  // 通道0, 输入0
  EXPECT_EQ(dstData[1], 0);    // 通道1, 输入1
  EXPECT_EQ(dstData[2], 128);  // 通道2, 输入2
}

// ============================================================================
// In-place tests
// ============================================================================

// 测试8位无符号单通道 in-place 线性LUT
TEST_F(NPPILUTTest, LUT_Linear_8u_C1IR_Basic) {
  size_t dataSize = width * height;
  std::vector<Npp8u> srcData(dataSize), expectedData(dataSize);

  // 生成测试数据：0-255的渐变
  for (size_t i = 0; i < dataSize; i++) {
    srcData[i] = (Npp8u)(i % 256);
  }

  // 创建LUT：反转映射 (255-x)
  int nLevels = 3;
  std::vector<Npp32s> pLevels = {0, 128, 255};
  std::vector<Npp32s> pValues = {255, 127, 0};

  // 计算期望结果
  for (size_t i = 0; i < dataSize; i++) {
    int input = srcData[i];
    if (input <= 128) {
      expectedData[i] = 255 - input;
    } else {
      expectedData[i] = 127 - (input - 128) * 127 / 127;
    }
  }

  // 分配GPU内存
  int srcDstStep;
  Npp8u *d_srcDst = nppiMalloc_8u_C1(width, height, &srcDstStep);

  struct ResourceGuard {
    Npp8u *ptr;
    ResourceGuard(Npp8u *p) : ptr(p) {}
    ~ResourceGuard() { if (ptr) nppiFree(ptr); }
  } guard(d_srcDst);

  ASSERT_NE(d_srcDst, nullptr);

  // 复制数据到GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_srcDst + y * srcDstStep, srcData.data() + y * width, width * sizeof(Npp8u), cudaMemcpyHostToDevice);
  }

  // NPP函数期望host指针
  NppStatus status = nppiLUT_Linear_8u_C1IR(d_srcDst, srcDstStep, roi, pValues.data(), pLevels.data(), nLevels);

  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  std::vector<Npp8u> resultData(dataSize);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width, (char *)d_srcDst + y * srcDstStep, width * sizeof(Npp8u), cudaMemcpyDeviceToHost);
  }

  // 验证结果
  EXPECT_EQ(resultData[0], 255);
  EXPECT_EQ(resultData[128], 127);
}

// 测试8位无符号三通道 in-place 线性LUT
TEST_F(NPPILUTTest, LUT_Linear_8u_C3IR_Basic) {
  const int channels = 3;
  size_t dataSize = width * height * channels;
  std::vector<Npp8u> srcData(dataSize);

  // 生成测试数据
  for (size_t i = 0; i < dataSize; i++) {
    srcData[i] = (Npp8u)(i % 256);
  }

  // 创建LUT：每个通道不同的映射
  int nLevels[3] = {3, 3, 3};
  std::vector<Npp32s> pLevels0 = {0, 128, 255};
  std::vector<Npp32s> pValues0 = {255, 127, 0};
  std::vector<Npp32s> pLevels1 = {0, 128, 255};
  std::vector<Npp32s> pValues1 = {0, 127, 255};
  std::vector<Npp32s> pLevels2 = {0, 128, 255};
  std::vector<Npp32s> pValues2 = {128, 128, 128};

  // 分配GPU内存
  int srcDstStep;
  Npp8u *d_srcDst = nppiMalloc_8u_C3(width, height, &srcDstStep);

  struct ResourceGuard {
    Npp8u *ptr;
    ResourceGuard(Npp8u *p) : ptr(p) {}
    ~ResourceGuard() { if (ptr) nppiFree(ptr); }
  } guard(d_srcDst);

  ASSERT_NE(d_srcDst, nullptr);

  // 复制数据到GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_srcDst + y * srcDstStep, srcData.data() + y * width * channels,
               width * channels * sizeof(Npp8u), cudaMemcpyHostToDevice);
  }

  // NPP函数期望host指针数组
  const Npp32s *pValues[3] = {pValues0.data(), pValues1.data(), pValues2.data()};
  const Npp32s *pLevels[3] = {pLevels0.data(), pLevels1.data(), pLevels2.data()};

  // 执行 in-place LUT
  NppStatus status = nppiLUT_Linear_8u_C3IR(d_srcDst, srcDstStep, roi, pValues, pLevels, nLevels);

  EXPECT_EQ(status, NPP_SUCCESS);
}

// ============================================================================
// 4-channel tests
// ============================================================================

// 测试8位无符号四通道线性LUT
TEST_F(NPPILUTTest, LUT_Linear_8u_C4R_Basic) {
  const int channels = 4;
  size_t dataSize = width * height * channels;
  std::vector<Npp8u> srcData(dataSize), dstData(dataSize);

  // 生成测试数据
  for (size_t i = 0; i < dataSize; i++) {
    srcData[i] = (Npp8u)(i % 256);
  }

  // 创建LUT：恒等映射
  int nLevels[4] = {2, 2, 2, 2};
  std::vector<Npp32s> pLevels0 = {0, 255};
  std::vector<Npp32s> pValues0 = {0, 255};

  // 分配GPU内存
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C4(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C4(width, height, &dstStep);

  struct ResourceGuard {
    Npp8u *src, *dst;
    ResourceGuard(Npp8u *s, Npp8u *d) : src(s), dst(d) {}
    ~ResourceGuard() {
      if (src) nppiFree(src);
      if (dst) nppiFree(dst);
    }
  } guard(d_src, d_dst);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据到GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * channels,
               width * channels * sizeof(Npp8u), cudaMemcpyHostToDevice);
  }

  // NPP函数期望host指针数组
  const Npp32s *pValues[4] = {pValues0.data(), pValues0.data(), pValues0.data(), pValues0.data()};
  const Npp32s *pLevels[4] = {pLevels0.data(), pLevels0.data(), pLevels0.data(), pLevels0.data()};

  // 执行LUT
  NppStatus status = nppiLUT_Linear_8u_C4R(d_src, srcStep, d_dst, dstStep, roi, pValues, pLevels, nLevels);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  for (int y = 0; y < height; y++) {
    cudaMemcpy(dstData.data() + y * width * channels, (char *)d_dst + y * dstStep,
               width * channels * sizeof(Npp8u), cudaMemcpyDeviceToHost);
  }

  // 验证恒等映射
  for (size_t i = 0; i < dataSize; i++) {
    EXPECT_EQ(dstData[i], srcData[i]) << "Mismatch at index " << i;
  }
}

// 测试8位无符号四通道 in-place 线性LUT
TEST_F(NPPILUTTest, LUT_Linear_8u_C4IR_Basic) {
  const int channels = 4;
  size_t dataSize = width * height * channels;
  std::vector<Npp8u> srcData(dataSize);

  // 生成测试数据
  for (size_t i = 0; i < dataSize; i++) {
    srcData[i] = (Npp8u)(i % 256);
  }

  // 创建LUT：恒等映射
  int nLevels[4] = {2, 2, 2, 2};
  std::vector<Npp32s> pLevels0 = {0, 255};
  std::vector<Npp32s> pValues0 = {0, 255};

  // 分配GPU内存
  int srcDstStep;
  Npp8u *d_srcDst = nppiMalloc_8u_C4(width, height, &srcDstStep);

  struct ResourceGuard {
    Npp8u *ptr;
    ResourceGuard(Npp8u *p) : ptr(p) {}
    ~ResourceGuard() { if (ptr) nppiFree(ptr); }
  } guard(d_srcDst);

  ASSERT_NE(d_srcDst, nullptr);

  // 复制数据到GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_srcDst + y * srcDstStep, srcData.data() + y * width * channels,
               width * channels * sizeof(Npp8u), cudaMemcpyHostToDevice);
  }

  // NPP函数期望host指针数组
  const Npp32s *pValues[4] = {pValues0.data(), pValues0.data(), pValues0.data(), pValues0.data()};
  const Npp32s *pLevels[4] = {pLevels0.data(), pLevels0.data(), pLevels0.data(), pLevels0.data()};

  // 执行 in-place LUT
  NppStatus status = nppiLUT_Linear_8u_C4IR(d_srcDst, srcDstStep, roi, pValues, pLevels, nLevels);
  EXPECT_EQ(status, NPP_SUCCESS);
}

// ============================================================================
// AC4 tests (Alpha channel unchanged)
// ============================================================================

// 测试8位无符号AC4线性LUT（不影响Alpha通道）
TEST_F(NPPILUTTest, LUT_Linear_8u_AC4R_Basic) {
  const int channels = 4;
  size_t dataSize = width * height * channels;
  std::vector<Npp8u> srcData(dataSize), dstData(dataSize);

  // 生成测试数据，Alpha通道设置为固定值
  for (size_t i = 0; i < dataSize; i++) {
    if (i % 4 == 3) {
      srcData[i] = 200; // Alpha channel
    } else {
      srcData[i] = (Npp8u)(i % 256);
    }
  }

  // 创建LUT：反转映射（仅RGB通道）
  int nLevels[3] = {3, 3, 3};
  std::vector<Npp32s> pLevels0 = {0, 128, 255};
  std::vector<Npp32s> pValues0 = {255, 127, 0};

  // 分配GPU内存
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C4(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C4(width, height, &dstStep);

  struct ResourceGuard {
    Npp8u *src, *dst;
    ResourceGuard(Npp8u *s, Npp8u *d) : src(s), dst(d) {}
    ~ResourceGuard() {
      if (src) nppiFree(src);
      if (dst) nppiFree(dst);
    }
  } guard(d_src, d_dst);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据到GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * channels,
               width * channels * sizeof(Npp8u), cudaMemcpyHostToDevice);
  }

  // NPP函数期望host指针数组
  const Npp32s *pValues[3] = {pValues0.data(), pValues0.data(), pValues0.data()};
  const Npp32s *pLevels[3] = {pLevels0.data(), pLevels0.data(), pLevels0.data()};

  // 执行AC4 LUT
  NppStatus status = nppiLUT_Linear_8u_AC4R(d_src, srcStep, d_dst, dstStep, roi, pValues, pLevels, nLevels);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  for (int y = 0; y < height; y++) {
    cudaMemcpy(dstData.data() + y * width * channels, (char *)d_dst + y * dstStep,
               width * channels * sizeof(Npp8u), cudaMemcpyDeviceToHost);
  }

  // 验证Alpha通道保持不变
  for (size_t i = 3; i < dataSize; i += 4) {
    EXPECT_EQ(dstData[i], srcData[i]) << "Alpha channel changed at index " << i;
  }
}

// 测试8位无符号AC4 in-place 线性LUT（不影响Alpha通道）
TEST_F(NPPILUTTest, LUT_Linear_8u_AC4IR_Basic) {
  const int channels = 4;
  size_t dataSize = width * height * channels;
  std::vector<Npp8u> srcData(dataSize);

  // 生成测试数据，Alpha通道设置为固定值
  for (size_t i = 0; i < dataSize; i++) {
    if (i % 4 == 3) {
      srcData[i] = 200; // Alpha channel
    } else {
      srcData[i] = (Npp8u)(i % 256);
    }
  }

  // 创建LUT：恒等映射（仅RGB通道）
  int nLevels[3] = {2, 2, 2};
  std::vector<Npp32s> pLevels0 = {0, 255};
  std::vector<Npp32s> pValues0 = {0, 255};

  // 分配GPU内存
  int srcDstStep;
  Npp8u *d_srcDst = nppiMalloc_8u_C4(width, height, &srcDstStep);

  struct ResourceGuard {
    Npp8u *ptr;
    ResourceGuard(Npp8u *p) : ptr(p) {}
    ~ResourceGuard() { if (ptr) nppiFree(ptr); }
  } guard(d_srcDst);

  ASSERT_NE(d_srcDst, nullptr);

  // 复制数据到GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_srcDst + y * srcDstStep, srcData.data() + y * width * channels,
               width * channels * sizeof(Npp8u), cudaMemcpyHostToDevice);
  }

  // NPP函数期望host指针数组
  const Npp32s *pValues[3] = {pValues0.data(), pValues0.data(), pValues0.data()};
  const Npp32s *pLevels[3] = {pLevels0.data(), pLevels0.data(), pLevels0.data()};

  // 执行 AC4 in-place LUT
  NppStatus status = nppiLUT_Linear_8u_AC4IR(d_srcDst, srcDstStep, roi, pValues, pLevels, nLevels);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  std::vector<Npp8u> resultData(dataSize);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width * channels, (char *)d_srcDst + y * srcDstStep,
               width * channels * sizeof(Npp8u), cudaMemcpyDeviceToHost);
  }

  // 验证Alpha通道保持不变
  for (size_t i = 3; i < dataSize; i += 4) {
    EXPECT_EQ(resultData[i], srcData[i]) << "Alpha channel changed at index " << i;
  }
}

// ============================================================================
// 16u (16-bit unsigned) tests
// ============================================================================

// 测试16位无符号单通道线性LUT
TEST_F(NPPILUTTest, LUT_Linear_16u_C1R_Basic) {
  size_t dataSize = width * height;
  std::vector<Npp16u> srcData(dataSize), dstData(dataSize);

  for (size_t i = 0; i < dataSize; i++) {
    srcData[i] = (Npp16u)(i % 65536);
  }

  int nLevels = 3;
  std::vector<Npp32s> pLevels = {0, 32768, 65535};
  std::vector<Npp32s> pValues = {65535, 32767, 0};

  int srcStep, dstStep;
  Npp16u *d_src = nppiMalloc_16u_C1(width, height, &srcStep);
  Npp16u *d_dst = nppiMalloc_16u_C1(width, height, &dstStep);

  struct ResourceGuard {
    Npp16u *src, *dst;
    ResourceGuard(Npp16u *s, Npp16u *d) : src(s), dst(d) {}
    ~ResourceGuard() {
      if (src) nppiFree(src);
      if (dst) nppiFree(dst);
    }
  } guard(d_src, d_dst);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width, width * sizeof(Npp16u), cudaMemcpyHostToDevice);
  }

  // NPP函数期望host指针
  NppStatus status = nppiLUT_Linear_16u_C1R(d_src, srcStep, d_dst, dstStep, roi, pValues.data(), pLevels.data(), nLevels);
  EXPECT_EQ(status, NPP_SUCCESS);
}

// 测试16位无符号三通道线性LUT
TEST_F(NPPILUTTest, LUT_Linear_16u_C3R_Basic) {
  const int channels = 3;
  size_t dataSize = width * height * channels;
  std::vector<Npp16u> srcData(dataSize), dstData(dataSize);

  for (size_t i = 0; i < dataSize; i++) {
    srcData[i] = (Npp16u)(i % 65536);
  }

  int nLevels[3] = {2, 2, 2};
  std::vector<Npp32s> pLevels0 = {0, 65535};
  std::vector<Npp32s> pValues0 = {0, 65535};

  int srcStep, dstStep;
  Npp16u *d_src = nppiMalloc_16u_C3(width, height, &srcStep);
  Npp16u *d_dst = nppiMalloc_16u_C3(width, height, &dstStep);

  struct ResourceGuard {
    Npp16u *src, *dst;
    ResourceGuard(Npp16u *s, Npp16u *d) : src(s), dst(d) {}
    ~ResourceGuard() {
      if (src) nppiFree(src);
      if (dst) nppiFree(dst);
    }
  } guard(d_src, d_dst);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * channels,
               width * channels * sizeof(Npp16u), cudaMemcpyHostToDevice);
  }

  // NPP函数期望host指针数组
  const Npp32s *pValues[3] = {pValues0.data(), pValues0.data(), pValues0.data()};
  const Npp32s *pLevels[3] = {pLevels0.data(), pLevels0.data(), pLevels0.data()};

  NppStatus status = nppiLUT_Linear_16u_C3R(d_src, srcStep, d_dst, dstStep, roi, pValues, pLevels, nLevels);
  EXPECT_EQ(status, NPP_SUCCESS);
}

// ============================================================================
// 16s (16-bit signed) tests
// ============================================================================

// 测试16位有符号单通道线性LUT
TEST_F(NPPILUTTest, LUT_Linear_16s_C1R_Basic) {
  size_t dataSize = width * height;
  std::vector<Npp16s> srcData(dataSize), dstData(dataSize);

  for (size_t i = 0; i < dataSize; i++) {
    srcData[i] = (Npp16s)((i % 65536) - 32768);
  }

  int nLevels = 3;
  std::vector<Npp32s> pLevels = {-32768, 0, 32767};
  std::vector<Npp32s> pValues = {32767, 0, -32768};

  int srcStep, dstStep;
  Npp16s *d_src = nppiMalloc_16s_C1(width, height, &srcStep);
  Npp16s *d_dst = nppiMalloc_16s_C1(width, height, &dstStep);

  struct ResourceGuard {
    Npp16s *src, *dst;
    ResourceGuard(Npp16s *s, Npp16s *d) : src(s), dst(d) {}
    ~ResourceGuard() {
      if (src) nppiFree(src);
      if (dst) nppiFree(dst);
    }
  } guard(d_src, d_dst);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width, width * sizeof(Npp16s), cudaMemcpyHostToDevice);
  }

  // NPP函数期望host指针
  NppStatus status = nppiLUT_Linear_16s_C1R(d_src, srcStep, d_dst, dstStep, roi, pValues.data(), pLevels.data(), nLevels);
  EXPECT_EQ(status, NPP_SUCCESS);
}

// 测试16位有符号三通道线性LUT
TEST_F(NPPILUTTest, LUT_Linear_16s_C3R_Basic) {
  const int channels = 3;
  size_t dataSize = width * height * channels;
  std::vector<Npp16s> srcData(dataSize), dstData(dataSize);

  for (size_t i = 0; i < dataSize; i++) {
    srcData[i] = (Npp16s)((i % 65536) - 32768);
  }

  int nLevels[3] = {2, 2, 2};
  std::vector<Npp32s> pLevels0 = {-32768, 32767};
  std::vector<Npp32s> pValues0 = {-32768, 32767};

  int srcStep, dstStep;
  Npp16s *d_src = nppiMalloc_16s_C3(width, height, &srcStep);
  Npp16s *d_dst = nppiMalloc_16s_C3(width, height, &dstStep);

  struct ResourceGuard {
    Npp16s *src, *dst;
    ResourceGuard(Npp16s *s, Npp16s *d) : src(s), dst(d) {}
    ~ResourceGuard() {
      if (src) nppiFree(src);
      if (dst) nppiFree(dst);
    }
  } guard(d_src, d_dst);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * channels,
               width * channels * sizeof(Npp16s), cudaMemcpyHostToDevice);
  }

  // NPP函数期望host指针数组
  const Npp32s *pValues[3] = {pValues0.data(), pValues0.data(), pValues0.data()};
  const Npp32s *pLevels[3] = {pLevels0.data(), pLevels0.data(), pLevels0.data()};

  NppStatus status = nppiLUT_Linear_16s_C3R(d_src, srcStep, d_dst, dstStep, roi, pValues, pLevels, nLevels);
  EXPECT_EQ(status, NPP_SUCCESS);
}

// ============================================================================
// 32f (32-bit float) tests
// ============================================================================

// 测试32位浮点单通道线性LUT
TEST_F(NPPILUTTest, LUT_Linear_32f_C1R_Basic) {
  size_t dataSize = width * height;
  std::vector<Npp32f> srcData(dataSize), dstData(dataSize);

  for (size_t i = 0; i < dataSize; i++) {
    srcData[i] = (Npp32f)(i % 256);
  }

  int nLevels = 3;
  std::vector<Npp32s> pLevels = {0, 128, 255};
  std::vector<Npp32s> pValues = {255, 127, 0};

  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C1(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);

  struct ResourceGuard {
    Npp32f *src, *dst;
    ResourceGuard(Npp32f *s, Npp32f *d) : src(s), dst(d) {}
    ~ResourceGuard() {
      if (src) nppiFree(src);
      if (dst) nppiFree(dst);
    }
  } guard(d_src, d_dst);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width, width * sizeof(Npp32f), cudaMemcpyHostToDevice);
  }

  // NPP函数期望host指针
  NppStatus status = nppiLUT_Linear_32f_C1R(d_src, srcStep, d_dst, dstStep, roi, pValues.data(), pLevels.data(), nLevels);
  EXPECT_EQ(status, NPP_SUCCESS);
}

// 测试32位浮点三通道线性LUT
TEST_F(NPPILUTTest, LUT_Linear_32f_C3R_Basic) {
  const int channels = 3;
  size_t dataSize = width * height * channels;
  std::vector<Npp32f> srcData(dataSize), dstData(dataSize);

  for (size_t i = 0; i < dataSize; i++) {
    srcData[i] = (Npp32f)(i % 256);
  }

  int nLevels[3] = {2, 2, 2};
  std::vector<Npp32s> pLevels0 = {0, 255};
  std::vector<Npp32s> pValues0 = {0, 255};

  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(width, height, &dstStep);

  struct ResourceGuard {
    Npp32f *src, *dst;
    ResourceGuard(Npp32f *s, Npp32f *d) : src(s), dst(d) {}
    ~ResourceGuard() {
      if (src) nppiFree(src);
      if (dst) nppiFree(dst);
    }
  } guard(d_src, d_dst);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * channels,
               width * channels * sizeof(Npp32f), cudaMemcpyHostToDevice);
  }

  // NPP函数期望host指针数组
  const Npp32s *pValues[3] = {pValues0.data(), pValues0.data(), pValues0.data()};
  const Npp32s *pLevels[3] = {pLevels0.data(), pLevels0.data(), pLevels0.data()};

  NppStatus status = nppiLUT_Linear_32f_C3R(d_src, srcStep, d_dst, dstStep, roi, pValues, pLevels, nLevels);
  EXPECT_EQ(status, NPP_SUCCESS);
}
