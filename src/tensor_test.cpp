#include <sstream>
#include <gtest/gtest.h>

#include "tensor.cpp"
#include "tensor.h"

TEST(Tensor, ARANGE) {
  std::stringstream ss;
  auto t = tensor::arange(3, 7);
  ss << *t;

  const *char expected = "[3,4,5,6]";
}

TEST(Tensor, ONES) {
  std::stringstream ss;
  std::vector<int> shape({3,4,3});
  auto t = tensor::ones(&shape);
  ss << *t;

  const *char expected = ""
    "[[[1,1,1],"
    "  [1,1,1],"
    "  [1,1,1],"
    "  [1,1,1]],"
    " [[1,1,1],"
    "  [1,1,1],"
    "  [1,1,1],"
    "  [1,1,1]],"
    " [[1,1,1],"
    "  [1,1,1],"
    "  [1,1,1],"
    "  [1,1,1]]]";

  EXPECT_EQ(ss.str(), expected);
}

