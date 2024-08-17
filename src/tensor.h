#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <memory>

namespace tensor {

  class Tensor {
    public:
    std::vector<int> shape;

    Tensor(std::vector<int> s) : shape(s) {}

    virtual std::string _str(int depth);
    virtual std::string str();
    virtual std::string shape_str();

    void print(std::ostream &os) {
      os << str();
    }

    friend std::ostream& operator<<(std::ostream &os, Tensor &t) {
      t.print(os);
      return os;
    }
  };

  std::unique_ptr<tensor::Tensor> arange(int start, int endExclusive);
  std::unique_ptr<tensor::Tensor> zeros(std::vector<int> &shape);
  std::unique_ptr<tensor::Tensor> ones(std::vector<int> &shape);

  namespace random {
    std::unique_ptr<tensor::Tensor> uniform(float min, float max, std::vector<int> &shape);
    std::unique_ptr<tensor::Tensor> uniform(std::vector<int> &shape);
  }
}

