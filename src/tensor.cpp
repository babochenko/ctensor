#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <random>

#include "tensor.h"

namespace tensor {

  std::vector<int> _tensor_deep(std::vector<std::unique_ptr<Tensor>> &v) {
    auto prev_shape = v[0]->shape;

    std::vector<int> shape;
    shape.reserve(prev_shape.size() + 1);

    shape.push_back(v.size());
    for (int i = 0; i < prev_shape.size(); i++) {
      shape.push_back(prev_shape[i]);
    }

    return shape;
  }

  class PTensor : public Tensor {
    using vec = std::variant<std::vector<float>, std::vector<std::unique_ptr<Tensor>>>;

    private:
    vec v;

    public:
    PTensor(std::vector<float> &v) : Tensor(std::vector<int>(1, v.size())) , v(v) {}
    PTensor(std::vector<std::unique_ptr<Tensor>> &v) : Tensor(_tensor_deep(v)), v(std::move(v)) {}

    std::string _str(int depth) override;
    std::string str() override;
  };

  std::string Tensor::_str(int depth) {
    return "Tensor()";
  }

  std::string Tensor::str() {
    return "Tensor()";
  }

  std::string Tensor::shape_str() {
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < shape.size(); i++) {
      ss << shape[i];
      if (i < shape.size() - 1) {
        ss << ", ";
      }
    }
    ss << ")";
    return ss.str();
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
    auto v = std::make_unique<std::vector<float>>(0);
    if (endExclusive > start) {
      for (float i=start; i < endExclusive; i++) {
        v->push_back(i + 0);
      }
    }
    return std::make_unique<PTensor>(*v);
  }

  std::unique_ptr<Tensor> _fill(std::vector<int> &shape, int depth, std::function<float()> fill) {
    if (depth == shape.size() - 1) {
      std::vector<float> v;
      v.reserve(shape[depth]);
      for (int i = 0; i < shape[depth]; i++) {
        v.push_back(fill());
      }

      return std::make_unique<PTensor>(v);
    }

    auto dim = shape[depth];
    std::vector<std::unique_ptr<Tensor>> data;
    data.reserve(dim);

    for (auto i = 0; i < dim; i++) {
      auto v = tensor::_fill(shape, depth + 1, fill);
      data.push_back(std::move(v));
    }
    return std::make_unique<PTensor>(data);
  }

  std::unique_ptr<Tensor> fill(std::vector<int> &shape, std::function<float()> fill) {
    if (shape.empty()) {
      throw std::invalid_argument("empty shape");
    }
    for (auto &val: shape) {
      if (val <= 0) {
        throw std::invalid_argument("non-positive shape");
      }
    }
    return tensor::_fill(shape, 0, fill);
  }

  std::unique_ptr<Tensor> zeros(std::vector<int> &shape) {
    return fill(shape, []() { return 0.0; });
  }

  std::unique_ptr<Tensor> ones(std::vector<int> &shape) {
    return fill(shape, []() { return 1.0; });
  }

  namespace random {
    std::unique_ptr<Tensor> uniform(float min, float max, std::vector<int> &shape) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> norm(min, max);

      return fill(shape, [&norm, &gen]() { return norm(gen); });
    }

    std::unique_ptr<Tensor> uniform(std::vector<int> &shape) {
      return uniform(0.0, 1.0, shape);
    }
  }
}

