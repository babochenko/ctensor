#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <random>

#include "tensor.h"

namespace tensor {

  std::vector<int> _shape(std::vector<std::shared_ptr<Tensor>> &v) {
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
    using vec = std::variant<std::vector<float>, std::vector<std::shared_ptr<Tensor>>>;

    public:
    PTensor(std::vector<float> &v) : Tensor(v, std::vector<int>(1, v.size())) {}
    PTensor(std::vector<std::shared_ptr<Tensor>> v) : Tensor(v, _shape(v)) {}

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

  std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> tensor1, std::shared_ptr<Tensor> tensor2) {
    if (tensor1->shape != tensor2->shape) {
      throw std::invalid_argument((std::stringstream()
            << "shape mismatch: " << tensor1->shape_str() << " and " << tensor2->shape_str()).str());
    }

    auto dim = tensor1->shape[0];

    return std::visit([dim](auto&& t1, auto&& t2) {
      using V1 = std::decay_t<decltype(t1[0])>;
      using V2 = std::decay_t<decltype(t2[0])>;
      if constexpr (std::is_same_v<V1, std::shared_ptr<Tensor>> && std::is_same_v<V2, std::shared_ptr<Tensor>>) {
        std::vector<std::shared_ptr<Tensor>> data;
        data.reserve(dim);

        for (size_t i = 0; i < t1.size(); i++) {
          auto x = t1[i];
          auto sum = t1[i] + t2[i];
          data.push_back(sum);
        }

        return std::make_shared<PTensor>(data);

      } else if constexpr (std::is_same_v<V1, float> && std::is_same_v<V2, float>) {
        std::vector<float> data;
        data.reserve(dim);

        for (size_t i = 0; i < t1.size(); i++) {
          auto sum = t1[i] + t2[i];
          data.push_back(sum);
        }

        return std::make_shared<PTensor>(data);
      } else {
        std::vector<float> data;
        return std::make_shared<PTensor>(data);
      }
    }, tensor1->vector, tensor2->vector);
  }

  std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> tensor) {
    auto dim = tensor->shape[0];

    return std::visit([dim](auto&& t) {
      using V1 = std::decay_t<decltype(t[0])>;
      if constexpr (std::is_same_v<V1, std::shared_ptr<Tensor>>) {
        std::vector<std::shared_ptr<Tensor>> data;
        data.reserve(dim);

        for (size_t i = 0; i < t.size(); i++) {
          auto x = t[i];
          auto sum = -t[i];
          data.push_back(sum);
        }

        return std::make_shared<PTensor>(data);

      } else if constexpr (std::is_same_v<V1, float>) {
        std::vector<float> data;
        data.reserve(dim);

        for (size_t i = 0; i < t.size(); i++) {
          auto sum = -t[i];
          data.push_back(sum);
        }

        return std::make_shared<PTensor>(data);
      } else {
        std::vector<float> data;
        return std::make_shared<PTensor>(data);
      }
    }, tensor->vector);
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

        if constexpr (std::is_same_v<VecT, std::shared_ptr<Tensor>>) {
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
    }, vector);
    ss << "]";
    return ss.str();
  } 

  std::string PTensor::str() {
    return _str(0);
  }

  std::shared_ptr<Tensor> arange(int start, int endExclusive) {
    auto v = std::make_shared<std::vector<float>>(0);
    if (endExclusive > start) {
      for (float i=start; i < endExclusive; i++) {
        v->push_back(i + 0);
      }
    }
    return std::make_shared<PTensor>(*v);
  }

  std::shared_ptr<Tensor> _fill(std::vector<int> &shape, int depth, std::function<float()> fill) {
    if (depth == shape.size() - 1) {
      std::vector<float> v;
      v.reserve(shape[depth]);
      for (int i = 0; i < shape[depth]; i++) {
        v.push_back(fill());
      }

      return std::make_shared<PTensor>(v);
    }

    auto dim = shape[depth];
    std::vector<std::shared_ptr<Tensor>> data;
    data.reserve(dim);

    for (auto i = 0; i < dim; i++) {
      auto v = tensor::_fill(shape, depth + 1, fill);
      data.push_back(v);
    }
    return std::make_shared<PTensor>(data);
  }

  std::shared_ptr<Tensor> fill(std::vector<int> &shape, std::function<float()> fill) {
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

  std::shared_ptr<Tensor> zeros(std::vector<int> &shape) {
    return fill(shape, []() { return 0.0; });
  }

  std::shared_ptr<Tensor> ones(std::vector<int> &shape) {
    return fill(shape, []() { return 1.0; });
  }

  namespace random {
    std::shared_ptr<Tensor> uniform(float min, float max, std::vector<int> &shape) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> norm(min, max);

      return fill(shape, [&norm, &gen]() { return norm(gen); });
    }

    std::shared_ptr<Tensor> uniform(std::vector<int> &shape) {
      return uniform(0.0, 1.0, shape);
    }
  }
}

