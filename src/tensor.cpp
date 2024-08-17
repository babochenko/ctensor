#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <stdexcept>

#include "tensor.h"

namespace tensor {

  class PTensor : public Tensor {
    using vec = std::variant<std::vector<int>, std::vector<std::unique_ptr<Tensor>>>;

    private:
    vec v;

    public:
    PTensor(std::vector<int> v) : v(v) {}
    PTensor(std::vector<std::unique_ptr<Tensor>> v) : v(std::move(v)) {}
    std::string _str(int depth) override;
    std::string str() override;
  };

  std::string Tensor::_str(int depth) {
    return "Tensor()";
  }

  std::string Tensor::str() {
    return "Tensor()";
  }

  std::ostream& operator<<(std::ostream &os, PTensor &ts) {
    os << ts.str();
    return os;
  }

  std::string PTensor::_str(int depth) {
    std::stringstream ss;
    ss << "[";
    std::visit([&ss, depth](auto&& vec) {
      for (auto i = 0; i < vec.size(); i++) {
        using VecT = std::decay_t<decltype(vec[i])>;

        if constexpr (std::is_same_v<VecT, std::unique_ptr<Tensor>>) {
          for (int j = 0; j < depth + 1; j++) {
            if (i > 0) {
              ss << " ";
            }
          }

          ss << vec[i]->_str(depth + 1);
          if (i < vec.size() - 1) {
            ss << "," << std::endl;
          }
        } else {
          ss << vec[i];
          if (i < vec.size() - 1) {
            ss << ",";
          }
        }
      }
    }, v);
    ss << "]";
    return ss.str();
  } 

  std::string PTensor::str() {
    return _str(0);
  }

  std::unique_ptr<Tensor> arange(int start, int endExclusive) {
    auto v = std::make_unique<std::vector<int> >(0);
    if (endExclusive > start) {
      for (auto i=start; i < endExclusive; i++) {
        v->push_back(i);
      }
    }
    return std::make_unique<PTensor>(*v);
  }

  std::unique_ptr<Tensor> _zeros_like(std::vector<int> &shape, int depth) {
    if (depth == shape.size() - 1) {
      return std::make_unique<PTensor>(std::vector<int>(shape[depth], 0));
    }

    auto dim = shape[depth];
    std::vector<std::unique_ptr<Tensor>> data;
    data.reserve(dim);

    for (auto i = 0; i < dim; i++) {
      auto v = tensor::_zeros_like(shape, depth + 1);
      data.push_back(std::move(v));
    }
    return std::make_unique<PTensor>(std::move(data));
  }

  std::unique_ptr<Tensor> zeros_like(std::vector<int> &shape) {
    if (shape.empty()) {
      throw std::invalid_argument("empty shape");
    }
    for (auto &val: shape) {
      if (val <= 0) {
        throw std::invalid_argument("non-positive shape");
      }
    }
    return tensor::_zeros_like(shape, 0);
  }
}

