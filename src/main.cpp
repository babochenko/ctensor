#include <iostream>
#include <memory>

#include "tensor.h"

int main() {

  auto s1 = std::vector<int>({2,4});
  auto t1 = tensor::ones(s1);

  auto s2 = std::vector<int>({4,2});
  auto t2 = tensor::ones(s2);

  auto res = t1->mul(t2);
  std::cout << *t1 << std::endl;

  return 0;
}

