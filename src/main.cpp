#include <iostream>
#include <memory>

#include "tensor.h"

int main() {

  auto s1 = std::vector<int>({3,4});
  auto t1 = tensor::ones(s1);

  auto s2 = std::vector<int>({4,3});
  auto t2 = tensor::ones(s2);

  auto res = t1->mul(t2);
  std::cout << *res << std::endl;

  return 0;
}

