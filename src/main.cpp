#include <iostream>
#include <memory>

#include "tensor.h"

int main() {
  auto t = tensor::arange(3, 7);
  std::cout << *t << std::endl;

  std::vector<int> shape({3,3,3});
  auto t1 = tensor::zeros_like(shape);
  std::cout << *t1 << std::endl;

  return 0;
}

