#include <iostream>
#include <memory>

#include "tensor.h"

int main() {

  auto s1 = std::vector<int>({10,10});
  auto t1 = tensor::random::uniform(s1);

  auto s2 = std::vector<int>({10,10});
  auto t2 = tensor::random::uniform(s2);

  auto res = t1->mul(t2);
  std::cout << *t1 << std::endl;
  std::cout << *t2 << std::endl;
  std::cout << *res << std::endl;
  std::cout << *(t1+t2) << std::endl;

  return 0;
}

