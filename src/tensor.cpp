#include <iostream>
#include <string>
#include <sstream>
#include <type_traits>
#include <vector>
#include <memory>
#include <stdexcept>
#include <random>

#include "tensor.h"

namespace tensor {

  using TNSR = std::shared_ptr<Tensor>;
  using P_VEC = std::vector<TNSR>;
  using V_VEC = std::vector<float>;
  using SHAPE = std::vector<int>;

  SHAPE _shape(P_VEC &v) {
    auto prev_shape = v[0]->shape;

    SHAPE shape;
    shape.reserve(prev_shape.size() + 1);

    shape.push_back(v.size());
    for (int i = 0; i < prev_shape.size(); i++) {
      shape.push_back(prev_shape[i]);
    }

    return shape;
  }

  class PTensor : public Tensor {
    using vec = std::variant<V_VEC, P_VEC>;

    public:
    PTensor(std::vector<float> &v) : Tensor(v, std::vector<int>(1, v.size())) {}
    PTensor(std::vector<std::shared_ptr<Tensor>> v) : Tensor(v, _shape(v)) {}

    std::string _str(int depth) override;
    std::string str() override;
  };

  TNSR tnsr(V_VEC data) {
    return std::make_shared<PTensor>(data);
  }

  TNSR tnsr(P_VEC data) {
    return std::make_shared<PTensor>(data);
  }

  std::string Tensor::_str(int depth) {
    return "Tensor()";
  }

  std::string Tensor::str() {
    return "Tensor()";
  }

  template <typename T>
  std::ostream& operator<<(std::ostream &os, std::vector<T> &vec) {
    os << "(";
    for (size_t i = 0; i < vec.size(); i++) {
      os << vec[i];
      if (i < vec.size() - 1) {
        os << ", ";
      }
    }
    os << ")";
    return os;
  }

  TNSR op_visit_tensor_value(
    TNSR tensor1,
    float t2,
    std::function<TNSR(TNSR, float)> t,
    std::function<float(float, float)> v
  ) {
    auto dim = tensor1->shape[0];

    return std::visit([&t, &v, t2, dim](auto&& t1) {
      using V1 = std::decay_t<decltype(t1[0])>;
      if constexpr (std::is_same_v<V1, TNSR>) {
        P_VEC data;
        data.reserve(dim);

        for (size_t i = 0; i < t1.size(); i++) {
          auto x = t1[i];
          auto res = t(t1[i], t2);
          data.push_back(res);
        }

        return tnsr(data);

      } else if constexpr (std::is_same_v<V1, float>) {
        V_VEC data;
        data.reserve(dim);

        for (size_t i = 0; i < t1.size(); i++) {
          auto res = v(t1[i], t2);
          data.push_back(res);
        }

        return tnsr(data);
      } else {
        V_VEC data;
        return tnsr(data);
      }
    }, tensor1->vector);
  }

  TNSR op_visit_tensors(
    TNSR tensor1,
    TNSR tensor2,
    std::function<TNSR(TNSR, TNSR)> t,
    std::function<float(float, float)> v
  ) {

    if (tensor1->shape != tensor2->shape) {
      std::stringstream s;
      s << "shape mismatch: " << tensor1->shape << " and " << tensor2->shape;
      throw std::invalid_argument(s.str());
    }

    auto dim = tensor1->shape[0];

    return std::visit([&t, &v, dim](auto&& t1, auto&& t2) {
      using V1 = std::decay_t<decltype(t1[0])>;
      using V2 = std::decay_t<decltype(t2[0])>;
      if constexpr (std::is_same_v<V1, TNSR> && std::is_same_v<V2, TNSR>) {
        P_VEC data;
        data.reserve(dim);

        for (size_t i = 0; i < t1.size(); i++) {
          auto x = t1[i];
          auto res = t(t1[i], t2[i]);
          data.push_back(res);
        }

        return tnsr(data);

      } else if constexpr (std::is_same_v<V1, float> && std::is_same_v<V2, float>) {
        V_VEC data;
        data.reserve(dim);

        for (size_t i = 0; i < t1.size(); i++) {
          auto res = v(t1[i], t2[i]);
          data.push_back(res);
        }

        return tnsr(data);
      } else {
        V_VEC data;
        return tnsr(data);
      }
    }, tensor1->vector, tensor2->vector);
  }

  TNSR operator+(TNSR tensor1, TNSR tensor2) {
    return op_visit_tensors(tensor1, tensor2, [](auto t1, auto t2) { return t1 + t2; }, [](auto t1, auto t2) { return t1 + t2; });
  }

  TNSR operator+(TNSR tensor, float v) {
    return op_visit_tensor_value(tensor, v, [](auto t1, auto t2) { return t1 + t2; }, [](auto t1, auto t2) { return t1 + t2; });
  }

  TNSR operator+(float v, TNSR tensor) {
    return tensor + v;
  }

  TNSR operator-(TNSR tensor) {
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

        return tnsr(data);

      } else if constexpr (std::is_same_v<V1, float>) {
        V_VEC data;
        data.reserve(dim);

        for (size_t i = 0; i < t.size(); i++) {
          auto sum = -t[i];
          data.push_back(sum);
        }

        return tnsr(data);
      } else {
        V_VEC data;
        return tnsr(data);
      }
    }, tensor->vector);
  }

  TNSR operator-(TNSR tensor1, TNSR tensor2) {
    return tensor1 + (-tensor2);
  }

  TNSR operator-(TNSR tensor, float v) {
    return tensor + (-v);
  }

  TNSR operator-(float v, TNSR tensor) {
    return tensor - v;
  }

  TNSR operator*(TNSR tensor1, TNSR tensor2) {
    return op_visit_tensors(tensor1, tensor2, [](auto t1, auto t2) { return t1 * t2; }, [](auto t1, auto t2) { return t1 * t2; });
  }

  TNSR operator*(TNSR tensor, float v) {
    return op_visit_tensor_value(tensor, v, [](auto t1, auto t2) { return t1 * t2; }, [](auto t1, auto t2) { return t1 * t2; });
  }

  TNSR operator*(float v, TNSR tensor) {
    return tensor * v;
  }

  TNSR operator/(TNSR tensor1, TNSR tensor2) {
    return op_visit_tensors(tensor1, tensor2, [](auto t1, auto t2) { return t1 / t2; }, [](auto t1, auto t2) { return t1 / t2; });
  }

  TNSR operator/(TNSR tensor, float v) {
    return op_visit_tensor_value(tensor, v, [](auto t1, auto t2) { return t1 / t2; }, [](auto t1, auto t2) { return t1 / t2; });
  }

  TNSR operator/(float v, TNSR tensor) {
    return tensor / v;
  }

  TNSR Tensor::T() {
    return T(0, 1);
  }

  TNSR Tensor::flatten() {
    V_VEC flattened = std::visit([this](auto&& v) {
      using VEC = std::decay_t<decltype(v)>;
      if constexpr (std::is_same_v<VEC, V_VEC>) {
        return v;

      } else if constexpr (std::is_same_v<VEC, P_VEC>) {
        V_VEC new_vec;
        auto size = 1;
        for (auto n : shape) {
          size *= n;
        }
        new_vec.reserve(size);

        for (size_t i = 0; i < v.size(); i++) {
          TNSR v_flattened = v[i]->flatten();
          std::visit([&new_vec](auto&& vf) {
            using VEC = std::decay_t<decltype(vf)>;
            if constexpr (std::is_same_v<VEC, V_VEC>) {
              for (auto el : vf) {
                new_vec.push_back(el);
              }
            }
          }, v_flattened->vector);
        }
        return new_vec;

      } else {
        return V_VEC();
      }
    }, vector);

    return tnsr(flattened);
  }

  TNSR Tensor::T(int dim1, int dim2) {
    if (shape.size() != 2) {
      throw std::invalid_argument("can't transpose a non-2D matrix (for now)");
    }

    auto flattened = flatten()->vector;

    P_VEC cols;
    std::visit([&cols, this] (auto&& v) {
      using VEC = std::decay_t<decltype(v)>;
      if constexpr (std::is_same_v<VEC, V_VEC>) {
        cols.reserve(shape[1]);
        for (auto i = 0; i < shape[1]; i++) {
          auto col = V_VEC(shape[0], 0);

          for (auto j = 0; j < shape[0]; j++) {
            auto idx = i + shape[1]*j;
            col[j] = v[idx];
          }
          cols.push_back(tnsr(col));
        }
      }
    }, flattened);

    return tnsr(cols);
  }

  TNSR Tensor::mul(TNSR other) {
    auto t = other->T();
    if (shape != t->shape) {
      std::stringstream s;
      s << "shape mismatch: " << shape << " and " << other->shape;
      throw std::invalid_argument(s.str());
    }

    auto is = shape[0];
    auto js = shape[1];
    auto cols = P_VEC();
    cols.reserve(js);

    for (size_t i = 0; i < shape[0]; i++) {
      std::visit([i, is, js](auto&& c1, auto&& c2) {
        using V1 = std::decay_t<decltype(c1)>;
        using V2 = std::decay_t<decltype(c2)>;
        if constexpr (std::is_same_v<V1, P_VEC> && std::is_same_v<V2, P_VEC>) {
          auto row1 = c1[i];
          auto row2 = c2[i];

          auto col = V_VEC();
          col.reserve(is);

          std::visit([&col, i, js](auto&& r1, auto&& r2) {
            using V1 = std::decay_t<decltype(r1)>;
            using V2 = std::decay_t<decltype(r2)>;
            if constexpr (std::is_same_v<V1, V_VEC> && std::is_same_v<V2, V_VEC>) {
              auto sum = 0;
              for (size_t j = 0; j < js; j++) {
                sum += r1[j] * r2[j];
              }
              col[i] = sum;
            }
          }, row1->vector, row2->vector);
        }
      }, this->vector, t->vector);
    }

    return tnsr(cols);
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

        if constexpr (std::is_same_v<VecT, TNSR>) {
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

  TNSR arange(int start, int endExclusive) {
    auto v = std::make_shared<V_VEC>(0);
    if (endExclusive > start) {
      for (float i=start; i < endExclusive; i++) {
        v->push_back(i + 0);
      }
    }
    return tnsr(*v);
  }

  TNSR _fill(SHAPE &shape, int depth, std::function<float()> fill) {
    if (depth == shape.size() - 1) {
      V_VEC v;
      v.reserve(shape[depth]);
      for (int i = 0; i < shape[depth]; i++) {
        v.push_back(fill());
      }

      return tnsr(v);
    }

    auto dim = shape[depth];
    P_VEC data;
    data.reserve(dim);

    for (auto i = 0; i < dim; i++) {
      auto v = tensor::_fill(shape, depth + 1, fill);
      data.push_back(v);
    }
    return tnsr(data);
  }

  TNSR fill(SHAPE &shape, std::function<float()> fill) {
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

  TNSR zeros(SHAPE &shape) {
    return fill(shape, []() { return 0.0; });
  }

  TNSR ones(SHAPE &shape) {
    return fill(shape, []() { return 1.0; });
  }

  namespace random {
    TNSR uniform(float min, float max, SHAPE &shape) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> norm(min, max);

      return fill(shape, [&norm, &gen]() { return norm(gen); });
    }

    TNSR uniform(SHAPE &shape) {
      return uniform(0.0, 1.0, shape);
    }
  }
}

