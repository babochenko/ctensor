#include <iostream>

#include "tensor.h"

int main() {
  tensor::Tensor t = tensor::arange(0, 10);
  std::cout << t.toString() << std::endl;
  return 0;
}

