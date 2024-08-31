#include <gtest/gtest.h>
#include <string>

#include "tensor.cpp"
#include "tensor.h"

void test_arange(int from, int toExclusive, std::string expected) {
  std::stringstream ss;
  auto t = tensor::arange(from, toExclusive);
  ss << *t;

  EXPECT_EQ(ss.str(), expected);
}

TEST(arange, 0arg) {
  test_arange(0, 0, "[]");
}

TEST(arange, 1arg) {
  test_arange(0, 1, "[0]");
}

TEST(arange, Nargs) {
  test_arange(3, 7, "[3,4,5,6]");
}

void expect(tensor::TNSR tensor, std::string expected) {
  std::stringstream ss;
  ss << *tensor;

  EXPECT_EQ(ss.str(), expected);
}

TEST(Tensor, ones) {
  auto t = tensor::ones(tensor::Shape{3,4,3});

  expect(t, ""
    "[[[1,1,1],\n"
    "  [1,1,1],\n"
    "  [1,1,1],\n"
    "  [1,1,1]],\n"
    " [[1,1,1],\n"
    "  [1,1,1],\n"
    "  [1,1,1],\n"
    "  [1,1,1]],\n"
    " [[1,1,1],\n"
    "  [1,1,1],\n"
    "  [1,1,1],\n"
    "  [1,1,1]]]");
}

TEST(Tensor, negate) {
  auto t1 = tensor::arange(0, 5);
  auto res = -t1;
  expect(res, "[-0,-1,-2,-3,-4]");
}

TEST(Tensor, exp1) {
  auto t1 = tensor::arange(0, 3);
  auto res = t1->exp();
  expect(res, "[1,2.71828,7.38906]");
}

TEST(Tensor, exp2) {
  auto t1 = tensor::arange(0, 3);
  auto res = t1->exp();
  expect(res, "[1,2.71828,7.38906]");
}

TEST(Tensor, sum1) {
  auto t1 = tensor::arange(0, 5);
  auto t2 = tensor::arange(0, 5);
  auto sum = t1 + t2;
  expect(sum, "[0,2,4,6,8]");
}

TEST(Tensor, sum2) {
  auto t1 = tensor::ones(tensor::Shape{2,2});
  auto t2 = tensor::ones(tensor::Shape{2,2});
  auto sum = t1 + t2;
  expect(sum, ""
    "[[2,2],\n"
    " [2,2]]");
}

TEST(Tensor, mul1) {
  auto t1 = tensor::ones(tensor::Shape{2,2});
  auto mul = 0.5 * t1;
  expect(mul, ""
    "[[0.5,0.5],\n"
    " [0.5,0.5]]");
}

TEST(Tensor, mul2) {
  auto t1 = tensor::arange(0, 4, tensor::Shape{2,2});
  auto t2 = tensor::arange(0, 4, tensor::Shape{2,2});
  auto mul = t1->mul(t2);
  expect(mul, ""
    "[[2,3],\n"
    " [6,11]]");
}

TEST(Tensor, resize1) {
  auto t = tensor::arange(0, 4);
  auto r = t->resize(tensor::Shape{2,2});
  expect(r, ""
    "[[0,1],\n"
    " [2,3]]");
}

TEST(Tensor, resize2) {
  auto t = tensor::arange(0, 6, tensor::Shape{2,3});
  expect(t, ""
    "[[0,1,2],\n"
    " [3,4,5]]");
}

TEST(Tensor, resize3) {
  auto t = tensor::arange(0, 12, tensor::Shape{2,2,3});
  expect(t, ""
    "[[[0,1,2],\n"
    "  [3,4,5]],\n"
    " [[6,7,8],\n"
    "  [9,10,11]]]");
}

