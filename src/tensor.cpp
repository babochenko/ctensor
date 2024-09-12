#include <iostream>
#include <string>
#include <sstream>
#include <type_traits>
#include <vector>
#include <memory>
#include <stdexcept>
#include <random>

#include "tensor.h"
#include "backward.h"

namespace tensor {

  template <typename T1, typename T2>
  constexpr bool is_() {
    using V = std::decay_t<T1>;
    return std::is_same_v<V, T2>;
  } 

  using PARENTS = std::vector<TNSR>;

  class PTensor : public Tensor {
    private:
    PARENTS parents;

    protected:
    void _do_backward(TNSR prev) override;

    public:
    PTensor(V_VEC &v, BW _backward, PARENTS parents) : Tensor(v, Shape(1, v.size()), _backward), parents(parents) {}
    PTensor(P_VEC v, BW _backward, PARENTS parents) : Tensor(v, _shape(v), _backward), parents(parents) {}

    void backward() override;

    std::string _str(int depth) override;
    std::string str() override;
  };

  int product(const Shape &shape) {
    int result = 1;
    for (auto item: shape) {
      result *= item;
    }
    return result;
  }

  TNSR tnsr(V_VEC data) {
    auto bw = std::make_shared<tensor::backward::NoOpV>(data);
    PARENTS prnts{0};
    return std::make_shared<PTensor>(data, bw, prnts);
  }

  TNSR tnsr(P_VEC data) {
    auto bw = std::make_shared<tensor::backward::NoOp>(data);
    PARENTS prnts;
    return std::make_shared<PTensor>(data, bw, prnts);
  }

  TNSR tnsr(V_VEC data, BW bw) {
    PARENTS prnts;
    return std::make_shared<PTensor>(data, bw, prnts);
  }

  TNSR tnsr(P_VEC data, BW bw) {
    PARENTS prnts;
    return std::make_shared<PTensor>(data, bw, prnts);
  }

  TNSR tnsr(V_VEC data, BW bw, PARENTS prnts) {
    return std::make_shared<PTensor>(data, bw, prnts);
  }

  TNSR tnsr(P_VEC data, BW bw, PARENTS prnts) {
    return std::make_shared<PTensor>(data, bw, prnts);
  }

  void Tensor::_do_backward(TNSR prev) {
  }

  void Tensor::backward() {
  }

  std::string Tensor::_str(int depth) {
    return "Tensor()";
  }

  std::string Tensor::str() {
    return "Tensor()";
  }

  float Tensor::item() {
    TNSR t = shared_from_this();
    return std::visit([](auto&& v) {
      if constexpr (is_<decltype(v), V_VEC>()) {
        if (v.size() == 1) {
          return v[0];
        }
      }

      std::stringstream s;
      s << "Can't get item from Tensor";
      throw std::invalid_argument(s.str());
      return 0.0f;
    }, t->vector);
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

  void PTensor::_do_backward(TNSR prev) {
    if (this->_backward) {
      this->grad = this->_backward->backward() * prev;
      if (!this->grad) {
        throw std::runtime_error("Gradient not properly initialized.");
      }

      for (auto prnt: this->parents) {
        prnt->_do_backward(this->grad);
      }
    } else {
      throw std::runtime_error("Backward function not initialized.");
    }
  }

  void PTensor::backward() {
    _do_backward(arange(1, 2));
  }

  template <typename L, typename R, typename FN = std::conditional<std::is_same<L, P_VEC>::value, TNSR, float>>
  TNSR apply(int size, L &left, R right, std::function<FN(FN, R)> mapper) {
    L data;
    data.reserve(size);

    for (auto item: left) {
      data.push_back(mapper(item, right));
    }

    return tnsr(data);
  }

  TNSR op_visit_tensor_value(
    TNSR tensor1,
    float t2,
    std::function<TNSR(TNSR, float)> t,
    std::function<float(float, float)> v
  ) {
    auto dim = tensor1->shape[0];

    return std::visit([&t, &v, t2, dim](auto&& t1) {
      if constexpr (is_<decltype(t1), P_VEC>()) {
        return apply<P_VEC, float>(dim, t1, t2, t);
      } else if constexpr (is_<decltype(t1), V_VEC>()) {
        return apply<V_VEC, float>(dim, t1, t2, v);
      } else {
        return tnsr(V_VEC{});
      }
    }, tensor1->vector);
  }

  TNSR op_visit_tensor(TNSR tensor, std::function<float(float)> v) {
    auto dim = tensor->shape[0];

    return std::visit([&v, dim](auto&& t) {
      if constexpr (is_<decltype(t[0]), TNSR>()) {
        P_VEC data;
        data.reserve(dim);

        for (size_t i = 0; i < t.size(); i++) {
          auto result = op_visit_tensor(t[i], v);
          data.push_back(result);
        }

        return tnsr(data);

      } else if constexpr (is_<decltype(t[0]), float>()) {
        V_VEC data;
        data.reserve(dim);

        for (size_t i = 0; i < t.size(); i++) {
          auto result = v(t[i]);
          data.push_back(result);
        }

        return tnsr(data);
      } else {
        V_VEC data;
        return tnsr(data);
      }
    }, tensor->vector);
  }

  void op_visit_tensor_inplace(TNSR tensor, std::function<void(float)> v) {
    auto dim = tensor->shape[0];

    std::visit([&v, dim](auto&& t) {
      if constexpr (is_<decltype(t[0]), TNSR>()) {
        for (size_t i = 0; i < t.size(); i++) {
          op_visit_tensor_inplace(t[i], v);
        }

      } else if constexpr (is_<decltype(t[0]), float>()) {
        for (size_t i = 0; i < t.size(); i++) {
          v(t[i]);
        }
      }
    }, tensor->vector);
  }

  TNSR op_visit_tensors(
    TNSR &tensor1,
    TNSR &tensor2,
    std::function<TNSR(TNSR, TNSR)> t,
    std::function<float(float, float)> v
  ) {
    compare_shapes(tensor1->shape, tensor2->shape);

    auto dim = tensor1->shape[0];

    return std::visit([&t, &v, dim](auto&& t1, auto&& t2) {
      if constexpr (is_<decltype(t1[0]), TNSR>() && is_<decltype(t2[0]), TNSR>()) {
        P_VEC data;
        data.reserve(dim);

        for (size_t i = 0; i < t1.size(); i++) {
          auto x = t1[i];
          auto res = t(t1[i], t2[i]);
          data.push_back(res);
        }

        return tnsr(data);

      } else if constexpr (is_<decltype(t1[0]), float>() && is_<decltype(t2[0]), float>()) {
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
    return op_visit_tensors(tensor1, tensor2, std::plus<>(), std::plus<>());
  }

  TNSR operator+(TNSR tensor, float v) {
    return op_visit_tensor_value(tensor, v, std::plus<>(), std::plus<>());
  }

  TNSR operator+(float v, TNSR tensor) {
    return tensor + v;
  }

  TNSR operator-(TNSR tensor) {
    return op_visit_tensor(tensor, std::negate<>());
  }

  TNSR Backward::backward() {
  }

  TNSR Tensor::exp() {
    TNSR t = shared_from_this();
    return op_visit_tensor(t, std::function<float(float)>(std::expf));
  }

  TNSR Tensor::log() {
    TNSR t = shared_from_this();
    return op_visit_tensor(t, std::function<float(float)>(std::logf));
  }

  TNSR Tensor::sum() {
    TNSR t = shared_from_this();
    float sum = 0.0;
    op_visit_tensor_inplace(t, [&sum](float val) { sum += val; });

    BW bw = std::make_unique<tensor::backward::Sum>();
    PARENTS prnts{t};

    return tnsr(V_VEC{sum}, bw, prnts);
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
    if (tensor2->shape == Shape{1}) {
      return std::visit([&tensor1](auto&& vec) {
        float data;
        if constexpr (is_<decltype(vec), V_VEC>()) {
          data = vec[0];
        } else {
          data = vec[0]->item();
        }

        return op_visit_tensor_value(tensor1, data, std::multiplies<>(), std::multiplies<>());
      }, tensor2->vector);
    }

    return op_visit_tensors(tensor1, tensor2, std::multiplies<>(), std::multiplies<>());
  }

  TNSR operator*(TNSR tensor, float v) {
    return op_visit_tensor_value(tensor, v, std::multiplies<>(), std::multiplies<>());
  }

  TNSR operator*(float v, TNSR tensor) {
    return tensor * v;
  }

  TNSR operator/(TNSR tensor1, TNSR tensor2) {
    return op_visit_tensors(tensor1, tensor2, std::divides<>(), std::divides<>());
  }

  TNSR operator/(TNSR tensor, float v) {
    return op_visit_tensor_value(tensor, v, std::divides<>(), std::divides<>());
  }

  TNSR operator/(float v, TNSR tensor) {
    return tensor / v;
  }

  TNSR Tensor::T() {
    return T(0, 1);
  }

  using Vec = std::variant<V_VEC, P_VEC>;

  V_VEC _flatten(Vec vector, Shape shape) {
    return std::visit([](auto&& vec) {
      if constexpr (is_<decltype(vec), V_VEC>()) {
        return vec;

      } else {
        V_VEC flattened;

        for (auto child: vec) {
          auto child_fl = _flatten(child->vector, child->shape);
          flattened.insert(flattened.end(), child_fl.begin(), child_fl.end());
        }
        return flattened;
      }
    }, vector);
  }

  void _resize(TNSR target, int &i, V_VEC &source) {
    std::visit([&source, &i](auto&& vec) {
      if constexpr (is_<decltype(vec), P_VEC>()) {
        for (auto row : vec) {
          _resize(row, i, source);
        }

      } else if constexpr (is_<decltype(vec), V_VEC>()) {
        for (size_t j = 0; j < vec.size(); j++) {
          vec[j] = source[i];
          i++;
        }
      }
    }, target->vector);
  }

  TNSR Tensor::resize(const Shape &shape) {
    auto sz1 = product(shape);
    auto sz2 = product(this->shape);
    if (sz1 != sz2) {
      Shape shp = shape;
      std::stringstream s;
      s << "shape mismatch: " << sz2 << " and " << shp;
      throw std::invalid_argument(s.str());
    }

    auto flat = _flatten(this->vector, this->shape);
    auto i = 0;
    auto result = zeros(shape);
    _resize(result, i, flat);

    return result;
  }

  TNSR Tensor::flatten() {
    return tnsr(_flatten(this->vector, this->shape));
  }

  TNSR Tensor::T(int dim1, int dim2) {
    if (shape.size() != 2) {
      throw std::invalid_argument("can't transpose a non-2D matrix (for now)");
    }

    auto flattened = _flatten(this->vector, this->shape);

    P_VEC cols;
    cols.reserve(shape[1]);

    for (auto i = 0; i < shape[1]; i++) {
      auto col = V_VEC(shape[0], 0);

      for (auto j = 0; j < shape[0]; j++) {
        auto idx = i + shape[1]*j;
        col[j] = flattened[idx];
      }
      cols.push_back(tnsr(col));
    }

    return tnsr(cols);
  }

  TNSR Tensor::mul(TNSR other) {
    auto t = other->T();
    compare_shapes(shape, t->shape);

    auto is = shape[0];
    auto js = shape[1];
    auto cols = P_VEC();
    cols.reserve(is);

    for (auto i = 0; i < shape[0]; i++) {
      auto col = V_VEC(is, float{});

      for (auto k = 0; k < shape[0]; k++) {
        std::visit([&col, i, k, is, js](auto&& c1, auto&& c2) {
          if constexpr (is_<decltype(c1), P_VEC>() && is_<decltype(c2), P_VEC>()) {

            std::visit([&col, i, k, js](auto&& r1, auto&& r2) {
              if constexpr (is_<decltype(r1), V_VEC>() && is_<decltype(r2), V_VEC>()) {
                auto sum = 0.0;
                for (auto j = 0; j < js; j++) {
                  sum += r1[j] * r2[j];
                }
                col[k] = sum;
              } else {
                std::stringstream s;
                s << "invalid type: " << typeid(r1).name();
                throw std::invalid_argument(s.str());
              }
            }, c1[i]->vector, c2[k]->vector);

          }
        }, this->vector, t->vector);
      }
      cols.push_back(tnsr(col));
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
        if constexpr (is_<decltype(vec[i]), TNSR>()) {
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

  TNSR arange(int start, int endExclusive, const Shape &shape) {
    return arange(start, endExclusive)->resize(shape);
  }

  TNSR _fill(const Shape &shape, int depth, std::function<float()> fill) {
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

  TNSR fill(const Shape &shape, std::function<float()> fill) {
    if (shape.empty()) {
      throw std::invalid_argument("empty shape");
    }
    for (auto &val: shape) {
      if (val <= 0) {
        Shape shp = shape;
        std::stringstream s;
        s << "cannot create tensor of shape " << shp;
        throw std::invalid_argument(s.str());
      }
    }
    return tensor::_fill(shape, 0, fill);
  }

  TNSR zeros(const Shape &shape) {
    return fill(shape, []() { return 0.0; });
  }

  TNSR ones(const Shape &shape) {
    return fill(shape, []() { return 1.0; });
  }

  namespace random {
    TNSR uniform(float min, float max, const Shape &shape) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> norm(min, max);

      return fill(shape, [&norm, &gen]() { return norm(gen); });
    }

    TNSR uniform(const Shape &shape) {
      return uniform(0.0, 1.0, shape);
    }
  }

  void compare_shapes(Shape shape1, Shape shape2) {
    if (shape1 != shape2) {
      std::stringstream s;
      s << "shape mismatch: " << shape1 << " and " << shape2;
      throw std::invalid_argument(s.str());
    }
  }
}

