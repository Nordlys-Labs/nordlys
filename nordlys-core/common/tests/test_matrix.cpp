#include <gtest/gtest.h>

#include <limits>
#include <nordlys/common/matrix.hpp>

class MatrixTest : public ::testing::Test {};

TEST_F(MatrixTest, BasicConstruction) {
  Matrix<float> m(3, 4);
  EXPECT_EQ(m.rows(), 3);
  EXPECT_EQ(m.cols(), 4);
  EXPECT_EQ(m.size(), 12);
}

TEST_F(MatrixTest, PointerConstruction) {
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Matrix<float> m(2, 3, data);
  EXPECT_EQ(m.rows(), 2);
  EXPECT_EQ(m.cols(), 3);
  EXPECT_EQ(m(0, 0), 1.0f);
  EXPECT_EQ(m(1, 2), 6.0f);
}

TEST_F(MatrixTest, ElementAccess) {
  Matrix<float> m(2, 2);
  m(0, 0) = 1.0f;
  m(0, 1) = 2.0f;
  m(1, 0) = 3.0f;
  m(1, 1) = 4.0f;
  EXPECT_EQ(m(0, 0), 1.0f);
  EXPECT_EQ(m(1, 1), 4.0f);
}

TEST_F(MatrixTest, Resize) {
  Matrix<float> m(2, 2);
  m.resize(3, 4);
  EXPECT_EQ(m.rows(), 3);
  EXPECT_EQ(m.cols(), 4);
  EXPECT_EQ(m.size(), 12);
}

TEST_F(MatrixTest, ConstructorOverflowThrows) {
  EXPECT_THROW({ Matrix<float>(std::numeric_limits<size_t>::max(), 2); }, std::overflow_error);
}

TEST_F(MatrixTest, PointerConstructorOverflowThrows) {
  float data[1] = {1.0f};
  EXPECT_THROW(
      { Matrix<float>(std::numeric_limits<size_t>::max(), 2, data); }, std::overflow_error);
}

TEST_F(MatrixTest, ResizeOverflowThrows) {
  Matrix<float> m(2, 2);
  EXPECT_THROW({ m.resize(std::numeric_limits<size_t>::max(), 2); }, std::overflow_error);
}

TEST_F(MatrixTest, ZeroSizeAllowed) {
  Matrix<float> m1(0, 0);
  EXPECT_EQ(m1.size(), 0);

  Matrix<float> m2(0, 10);
  EXPECT_EQ(m2.size(), 0);

  Matrix<float> m3(10, 0);
  EXPECT_EQ(m3.size(), 0);
}

TEST_F(MatrixTest, DataPointer) {
  Matrix<float> m(2, 3);
  float* ptr = m.data();
  EXPECT_NE(ptr, nullptr);
  ptr[0] = 42.0f;
  EXPECT_EQ(m(0, 0), 42.0f);
}

TEST_F(MatrixTest, ConstDataPointer) {
  const Matrix<float> m(2, 3);
  const float* ptr = m.data();
  EXPECT_NE(ptr, nullptr);
}

TEST_F(MatrixTest, DoubleType) {
  Matrix<double> m(2, 2);
  m(0, 0) = 1.5;
  m(1, 1) = 2.5;
  EXPECT_DOUBLE_EQ(m(0, 0), 1.5);
  EXPECT_DOUBLE_EQ(m(1, 1), 2.5);
}

TEST_F(MatrixTest, DefaultConstruction) {
  Matrix<float> m;
  EXPECT_EQ(m.rows(), 0);
  EXPECT_EQ(m.cols(), 0);
  EXPECT_EQ(m.size(), 0);
}

TEST_F(MatrixTest, CopyConstructor) {
  Matrix<float> m1(2, 3);
  m1(0, 0) = 1.0f;
  m1(1, 2) = 5.0f;

  Matrix<float> m2(m1);
  EXPECT_EQ(m2.rows(), 2);
  EXPECT_EQ(m2.cols(), 3);
  EXPECT_EQ(m2(0, 0), 1.0f);
  EXPECT_EQ(m2(1, 2), 5.0f);
}

TEST_F(MatrixTest, MoveConstructor) {
  Matrix<float> m1(2, 3);
  m1(0, 0) = 42.0f;

  Matrix<float> m2(std::move(m1));
  EXPECT_EQ(m2.rows(), 2);
  EXPECT_EQ(m2.cols(), 3);
  EXPECT_EQ(m2(0, 0), 42.0f);
}

TEST_F(MatrixTest, CopyAssignment) {
  Matrix<float> m1(2, 3);
  m1(0, 0) = 10.0f;

  Matrix<float> m2;
  m2 = m1;
  EXPECT_EQ(m2.rows(), 2);
  EXPECT_EQ(m2.cols(), 3);
  EXPECT_EQ(m2(0, 0), 10.0f);
}

TEST_F(MatrixTest, MoveAssignment) {
  Matrix<float> m1(2, 3);
  m1(1, 1) = 99.0f;

  Matrix<float> m2;
  m2 = std::move(m1);
  EXPECT_EQ(m2.rows(), 2);
  EXPECT_EQ(m2.cols(), 3);
  EXPECT_EQ(m2(1, 1), 99.0f);
}
