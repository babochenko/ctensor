#include <functional>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <random>

#include "tensor.h"
#include "backward.h"
#include "functional.h"

namespace tensor {

  int product(const Shape &shape) {
    int result = 1;
    for (auto item: shape) {
      result *= item;
    }
    return result;
  }

  Tensor::Tensor(V_VEC v) : vector(v) {
    shape = _shape(v);
  }

  Tensor::Tensor(P_VEC v) : vector(v) {
    shape = _shape(v);
  }

  template <typename V>
  TNSR tnsr(V vec) {
    return tnsr(vec, tensor::backward::ones(vec));
  }

  template <typename V>
  TNSR tnsr(V vec, BW bw) {
    return tnsr(vec, tensor::backward::ones(vec), PARENTS{});
  }

  template <typename V>
  TNSR tnsr(V vec, BW bw, PARENTS parents) {
    auto t = std::make_shared<Tensor>(vec);
    t->set_backward(bw, parents);
    return t;
  }

  float Tensor::item() {
    TNSR t = shared_from_this();
    compare(t->shape, Shape{1}, "item() shapes");

    return std::visit([](auto&& v) {
      if constexpr (is_<decltype(v), P_VEC>()) {
        return v[0]->item();
      } else if constexpr (is_<decltype(v), V_VEC>()) {
        return v[0];
      }

      std::stringstream s;
      s << "Can't get item from Tensor";
      throw std::invalid_argument(s.str());
      return 0.0f;
    }, t->vector);
  }

  void Tensor::_do_backward(TNSR prev) {
    if (this->_backward) {
      this->grad = this->_backward(prev);
      if (!this->grad) {
        throw std::runtime_error("Gradient not properly initialized.");
      }

      for (auto prnt: this->_parents) {
        prnt->_do_backward(this->grad);
      }
    } else {
      throw std::runtime_error("Backward function not initialized.");
    }
  }

  void Tensor::backward() {
    _do_backward(arange(1, 2));
  }
  TNSR operator+(TNSR tensor1, TNSR tensor2) {
    return func::merge(
      tensor1,
      tensor2,
      std::plus<>(),
      tensor::backward::ones(tensor1),
      tensor::backward::ones(tensor2)
    );
  }

  TNSR operator+(TNSR tensor, float v) {
    return func::map(tensor, v, std::plus<>());
  }

  TNSR operator+(float v, TNSR tensor) {
    return tensor + v;
  }

  TNSR operator-(TNSR tensor) {
    return func::map(tensor, std::negate<>());
  }

  TNSR Tensor::exp() {
    auto t = shared_from_this();
    auto res = func::map(t, std::expf);

    PARENTS prnts{t};
    auto bw = tensor::backward::exp(res);
    res->set_backward(bw, prnts);
    return res;
  }

  TNSR Tensor::log() {
    TNSR t = shared_from_this();
    auto res = func::map(t, std::logf);

    PARENTS prnts{t};
    auto bw = tensor::backward::exp(res);
    res->set_backward(bw, prnts);
    return res;
  }

  TNSR Tensor::sum() {
    TNSR t = shared_from_this();
    float sum = 0.0;
    func::visit(t, [&sum](float val) { sum += val; });

    BW bw = tensor::backward::sum();
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
    if (tensor1->shape == Shape{1}) {
      return func::map(tensor2, tensor1->item(), std::multiplies<>());
    } else if (tensor2->shape == Shape{1}) {
      return func::map(tensor1, tensor2->item(), std::multiplies<>());
    }

    return func::merge(
      tensor1,
      tensor2,
      std::multiplies<>(),
      tensor::backward::mul(tensor2),
      tensor::backward::mul(tensor1)
    );
  }

  TNSR operator*(TNSR tensor, float v) {
    return func::map(tensor, v, std::multiplies<>());
  }

  TNSR operator*(float v, TNSR tensor) {
    return tensor * v;
  }

  TNSR operator/(TNSR tensor1, TNSR tensor2) {
    return func::merge(
        tensor1,
        tensor2,
        std::divides<>(),
        tensor::backward::ones(tensor1), // TODO fix
        tensor::backward::ones(tensor2));
  }

  TNSR operator/(TNSR tensor, float v) {
    return func::map(tensor, v, std::divides<>());
  }

  TNSR operator/(float v, TNSR tensor) {
    return tensor / v;
  }

  TNSR Tensor::T() {
    return T(0, 1);
  }

  using Vec = std::variant<V_VEC, P_VEC>;

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

  V_VEC _flatten(TNSR tensor) {
    auto flat = V_VEC{};
    func::visit(tensor, [&flat](float el) { flat.push_back(el); });
    return flat;
  }

  V_VEC _unflatten(Vec vector, Shape shape) {
    // TODO write
  }

  TNSR Tensor::flatten() {
    return tnsr(_flatten(shared_from_this()));
  }

  TNSR Tensor::resize(const Shape &shape) {
    auto sz1 = product(shape);
    auto sz2 = product(this->shape);
    if (sz1 != sz2) {
      Shape shp = shape;
      std::stringstream s;
      s << "shape product mismatch: " << sz2 << " and " << shp;
      throw std::invalid_argument(s.str());
    }

    auto flat = _flatten(shared_from_this());
    auto i = 0;
    auto result = zeros(shape);
    _resize(result, i, flat);

    return result;
  }

  TNSR Tensor::T(int dim1, int dim2) {
    if (shape.size() != 2) {
      throw std::invalid_argument("can't transpose a non-2D matrix (for now)");
    }

    auto flattened = _flatten(shared_from_this());

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

  float Tensor::dot(TNSR other) {
    TNSR t = shared_from_this();
    auto t_shape = t->shape;
    compare(t_shape, other->shape, "dot() shapes");

    int x = t_shape.size();
    int y = 1;
    compare(x, y, t_shape, other->shape, "dot() shape sizes");

    auto len = t_shape[t_shape.size() == 1 ? 0 : 1];

    return std::visit([&len](auto&& r1, auto&& r2) {
      if constexpr (is_<decltype(r1), V_VEC>() && is_<decltype(r2), V_VEC>()) {
        float sum = 0.0;
        for (auto j = 0; j < len; j++) {
          sum += r1[j] * r2[j];
        }
        return sum;
      } else {
        std::stringstream s;
        s << "invalid type: " << typeid(r1).name();
        throw std::invalid_argument(s.str());
        return 0.0f;
      }
    }, t->vector, other->vector);
  }

  TNSR Tensor::mul(TNSR other) {
    auto t = other->T();
    compare(shape[1], t->shape[1], shape, t->shape, "mul() shapes");

    auto cols = shape[0];
    auto rows = t->shape[0];

    auto result = V_VEC();
    result.reserve(cols * rows);

    std::visit([&](auto&& vec1, auto&& vec2) {
      using V1 = decltype(vec1);
      using V2 = decltype(vec2);
      if constexpr (is_<V1, P_VEC>() && is_<V2, P_VEC>()) {
        for (auto i = 0; i < cols; i++) {
          for (auto j = 0; j < rows; j++) {
            result.push_back(vec1[i]->dot(vec2[j]));
          }
        }
      }
    }, this->vector, t->vector);

    TNSR _this = shared_from_this();
    PARENTS prnts;

    _this->_backward = tensor::backward::matMulLeft(other);
    other->_backward = tensor::backward::matMulRight(_this);
    prnts.push_back(_this);
    prnts.push_back(other);

    auto res = tnsr(result, tensor::backward::ones(_this), prnts);
    return res->resize(Shape{cols, rows});
  }

  std::ostream& operator<<(std::ostream &os, Tensor &ts) {
    os << ts.str();
    return os;
  }

  std::string Tensor::_str(int depth) {
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

  std::string Tensor::str() {
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

  template <typename Op>
  TNSR _fill(const Shape &shape, int depth, Op fill) {
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

  template <typename Op>
  TNSR fill(const Shape &shape, Op fill) {
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

  TNSR Tensor::pow(int power) {
    auto t = shared_from_this();
    return func::map(t, power + 0.0, std::plus<>());
  }

} // namespace tensor

