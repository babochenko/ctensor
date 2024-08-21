#include <iostream>
#include <memory>

#include "tensor.h"

int main() {

  auto t1 = tensor::random::uniform(tensor::Shape{10,10});
  auto t2 = tensor::random::uniform(tensor::Shape{10,10});

  auto res = t1->mul(t2);
  std::cout << *t1 << std::endl;
  std::cout << *t2 << std::endl;
  std::cout << *res << std::endl;
  std::cout << *(t1+t2) << std::endl;

  return 0;
}

