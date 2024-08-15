#include <iostream>

#include <iterator>
#include <string>
#include <sstream>
#include <vector>
#include <memory>

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
    ss << "Tensor(";
    for (auto i = 0; i < ts.size(); i++) {
      ss << ts.at(i);
      if (i < ts.size() - 1) {
        ss << ",";
      }
    }
    ss << ")";
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
}

