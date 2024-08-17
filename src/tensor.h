#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <memory>

namespace tensor {

  class Tensor {
    public:
    virtual std::string _str(int depth);
    virtual std::string str();

    void print(std::ostream &os) {
      os << str();
    }

    friend std::ostream& operator<<(std::ostream &os, Tensor &t) {
      t.print(os);
      return os;
    }
  };

  std::unique_ptr<tensor::Tensor> arange(int start, int endExclusive);
  std::unique_ptr<tensor::Tensor> zeros_like(std::vector<int> &shape);

}

