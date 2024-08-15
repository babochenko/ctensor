#include <iostream>

#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <stdexcept>

#include "tensor.h"

namespace tensor {

  template <typename T>
  class PTensor : public Tensor {
    private:
    std::vector<T> ts;

    public:
    PTensor(std::vector<T> ts) : Tensor(), ts(ts) {}
    std::string str() override;
  };

  std::string Tensor::str() {
    return "Tensor()";
  }

  template <typename T>
  std::ostream& operator<<(std::ostream &os, PTensor<T> &ts) {
    os << ts.str();
    return os;
  }

  template <typename T>
  std::string PTensor<T>::str() {
    std::stringstream ss;
    ss << "[";
    for (auto i = 0; i < ts.size(); i++) {
      ss << ts.at(i);
      if (i < ts.size() - 1) {
        ss << ",";
      }
    }
    ss << "]";
    return ss.str();
  } 

  std::unique_ptr<Tensor> arange(int start, int endExclusive) {
    auto v = std::make_unique<std::vector<int> >(0);
    if (endExclusive > start) {
      for (auto i=start; i < endExclusive; i++) {
        v->push_back(i);
      }
    }
    return std::make_unique<PTensor<int> >(*v);
  }

  template <typename T>
  std::unique_ptr<tensor::PTensor<T>> _zeros_like(std::vector<int> &shape, int depth) {
    if (depth == shape.size() - 1) {
      return std::make_unique<PTensor<T>>(std::vector<T>(shape[depth], T{}));
    }

    auto size = shape[depth];
    std::vector<std::unique_ptr<tensor::PTensor<T>>> data(size);

    for (auto i = 0; i < size; i++) {
      auto v = tensor::_zeros_like<T>(shape, size + 1);
      data.push_back(v);
    }
    return std::make_unique<PTensor<std::unique_ptr<PTensor<T>>>>(std::move(data));
  }

  std::unique_ptr<tensor::Tensor> zeros_like(std::vector<int> &shape) {
    if (shape.empty()) {
      throw std::invalid_argument("empty shape");
    }
    for (auto &val: shape) {
      if (val <= 0) {
        throw std::invalid_argument("non-positive shape");
      }
    }
    return tensor::_zeros_like<int>(shape, 0);
  }
}

