#include <iostream>
#include <memory>

#include "tensor.h"

int main() {

  std::vector<int> shape({3,4,3});

  auto t1 = tensor::ones(shape);
  auto t2 = tensor::ones(shape);
  std::cout << *(t1 + t2) << std::endl;

  return 0;
}

