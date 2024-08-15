#include <iostream>
#include <memory>

#include "tensor.h"

int main() {
  std::unique_ptr<tensor::Tensor> t = tensor::arange(3, 7);
  std::cout << *t << std::endl;
  return 0;
}

