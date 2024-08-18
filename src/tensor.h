#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <memory>

namespace tensor {

  class Tensor {
    using vec = std::variant<std::vector<float>, std::vector<std::shared_ptr<Tensor>>>;

    public:
    vec vector;
    std::vector<int> shape;

    Tensor(
        std::vector<float> &v,
        std::vector<int> s
    ) : shape(s) , vector(v) {}

    Tensor(
        std::vector<std::shared_ptr<Tensor>> v,
        std::vector<int> s
    ) : shape(s), vector(v) {}

    std::shared_ptr<Tensor> flatten();
    std::shared_ptr<Tensor> T();
    std::shared_ptr<Tensor> T(int dim1, int dim2);

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

  std::shared_ptr<tensor::Tensor> arange(int start, int endExclusive);
  std::shared_ptr<tensor::Tensor> zeros(std::vector<int> &shape);
  std::shared_ptr<tensor::Tensor> ones(std::vector<int> &shape);

  namespace random {
    std::shared_ptr<tensor::Tensor> uniform(float min, float max, std::vector<int> &shape);
    std::shared_ptr<tensor::Tensor> uniform(std::vector<int> &shape);
  }

  std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> tensor1, std::shared_ptr<Tensor> tensor2);
  std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> tensor, float v);
  std::shared_ptr<Tensor> operator+(float v, std::shared_ptr<Tensor> tensor);

  std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> tensor);

  std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> tensor1, std::shared_ptr<Tensor> tensor2);
  std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> tensor, float v);
  std::shared_ptr<Tensor> operator-(float v, std::shared_ptr<Tensor> tensor);

  std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> tensor1, std::shared_ptr<Tensor> tensor2);
  std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> tensor, float v);
  std::shared_ptr<Tensor> operator*(float v, std::shared_ptr<Tensor> tensor);

  std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> tensor1, std::shared_ptr<Tensor> tensor2);
  std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> tensor, float v);
  std::shared_ptr<Tensor> operator/(float v, std::shared_ptr<Tensor> tensor);
}

