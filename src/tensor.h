#pragma once

#include <type_traits>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <memory>

namespace tensor {

  template <typename T1, typename T2>
  constexpr bool is_() {
    using V = std::decay_t<T1>;
    return std::is_same_v<V, T2>;
  } 

  class Tensor;

  using Shape = std::vector<int>;
  using TNSR = std::shared_ptr<Tensor>;
  using P_VEC = std::vector<TNSR>;
  using V_VEC = std::vector<float>;
  using PARENTS = std::vector<TNSR>;

  class Backward {
    public:
    virtual TNSR backward(TNSR prev);
  };

  using BW = std::shared_ptr<Backward>;

  class Tensor : public std::enable_shared_from_this<Tensor> {
    using Vec = std::variant<V_VEC, P_VEC>;

    private:
    BW _backward;
    PARENTS _parents;

    public:
    Vec vector;
    Shape shape;
    TNSR grad;

    Tensor(V_VEC vec);
    Tensor(P_VEC vec);

    void set_backward(BW bw) {
      this->_backward = bw;
    }

    void set_backward(BW bw, PARENTS parents) {
      this->_backward = bw;
      this->_parents = parents;
    }

    template <typename FP, typename FV>
    auto visit(FP fp, FV fv) {
      std::visit([&](auto&& vector) {
        using ELMNT = decltype(vector[0]);
        if constexpr (is_<ELMNT, tensor::TNSR>()) {
          return fp(vector);
        } else if constexpr (is_<ELMNT, float>()) {
          return fv(vector);
        }
      }, this->vector);
    }

    TNSR resize(const Shape &shape);
    TNSR flatten();
    TNSR T();
    TNSR T(int dim1, int dim2);
    TNSR mul(TNSR other);
    float dot(TNSR other);

    TNSR exp();
    TNSR log();
    TNSR sum();
    TNSR pow(int power);
    float item();

    void _do_backward(TNSR prev);
    void backward();

    virtual std::string _str(int depth);
    virtual std::string str();
    void print(std::ostream &os) { os << str(); }

    friend std::ostream& operator<<(std::ostream &os, Tensor &t);
  };

  template <typename V>
  Shape _shape(V v);

  template <>
  inline Shape _shape(TNSR v) {
    return v->shape;
  }

  template <>
  inline Shape _shape(V_VEC v) {
    return Shape{1, static_cast<int>(v.size())};
  }

  template <>
  inline Shape _shape(P_VEC v) {
    int prev_size = v.size();
    Shape prev = v[0]->shape;

    Shape shape = { prev_size };
    shape.insert(shape.end(), prev.begin(), prev.end());

    return shape;
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

  template <typename V> TNSR tnsr(V vec);
  template <typename V> TNSR tnsr(V vec, BW bw, PARENTS parents);
  template <typename V> TNSR tnsr(V vec, BW bw, PARENTS parents);

  TNSR arange(int start, int endExclusive);
  TNSR arange(int start, int endExclusive, const Shape &shape);
  TNSR zeros(const Shape &shape);
  TNSR ones(const Shape &shape);

  namespace random {
    TNSR uniform(float min, float max, Shape &shape);
    TNSR uniform(const Shape &shape);
  }

  TNSR operator+(TNSR tensor1, TNSR tensor2);
  TNSR operator+(TNSR tensor, float v);
  TNSR operator+(float v, TNSR tensor);

  TNSR operator-(TNSR tensor);

  TNSR operator-(TNSR tensor1, TNSR tensor2);
  TNSR operator-(TNSR tensor, float v);
  TNSR operator-(float v, TNSR tensor);

  TNSR operator*(TNSR tensor1, TNSR tensor2);
  TNSR operator*(TNSR tensor, float v);
  TNSR operator*(float v, TNSR tensor);

  TNSR operator/(TNSR tensor1, TNSR tensor2);
  TNSR operator/(TNSR tensor, float v);
  TNSR operator/(float v, TNSR tensor);

  void compare_shapes(Shape shape1, Shape shape2);
  void compare_shapes(Shape shape1, Shape shape2, size_t idx);
}

