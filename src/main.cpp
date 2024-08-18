#include <iostream>
#include <memory>

#include "tensor.h"

int main() {

  std::vector<int> shape({2,4});

  auto t = tensor::ones(shape);
  std::cout << *t << std::endl;

  auto t1 = t->T();
  std::cout << *t1 << std::endl;

  return 0;
}

