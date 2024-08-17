#include <iostream>
#include <memory>

#include "tensor.h"

int main() {
  auto t = tensor::arange(3, 7);
  std::cout << *t << std::endl;

  std::vector<int> shape({3,3,3});

  t = tensor::ones(shape);
  std::cout << *t << std::endl;

  t = tensor::random::uniform(0.0, 3.0, shape);
  std::cout << *t << std::endl;

  return 0;
}

