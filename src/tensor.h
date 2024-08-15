#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <memory>

namespace tensor {

  class Tensor {
    public:
    void print(std::ostream &os) {
      os << str();
    }

    virtual std::string str();

    friend std::ostream& operator<<(std::ostream &os, Tensor &t) {
      t.print(os);
      return os;
    }
  };

  std::unique_ptr<tensor::Tensor> arange(int start, int endExclusive);

}

