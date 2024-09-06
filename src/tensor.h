#pragma once

#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <memory>

namespace tensor {

  class Tensor;

  using Shape = std::vector<int>;
  using TNSR = std::shared_ptr<Tensor>;
  using P_VEC = std::vector<TNSR>;
  using V_VEC = std::vector<float>;
  using BW = std::function<TNSR()>;

  class Tensor : public std::enable_shared_from_this<Tensor> {
    using Vec = std::variant<V_VEC, P_VEC>;

    private:
    BW _backward;

    public:
    Vec vector;
    Shape shape;
    TNSR grad;

    Tensor(V_VEC &v, Shape s, BW _backward) : shape(s), vector(v), _backward(_backward) {}
    Tensor(P_VEC v, Shape s, BW _backward) : shape(s), vector(v), _backward(_backward) {}

    TNSR resize(const Shape &shape);
    TNSR flatten();
    TNSR T();
    TNSR T(int dim1, int dim2);
    TNSR mul(TNSR other);

    TNSR exp();
    TNSR log();
    TNSR sum();
    float item();

    void backward();

    virtual std::string _str(int depth);
    virtual std::string str();
    void print(std::ostream &os) { os << str(); }

    friend std::ostream& operator<<(std::ostream &os, Tensor &t) {
      t.print(os);
      return os;
    }
  };

  TNSR tnsr(V_VEC data);
  TNSR tnsr(P_VEC data);

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
}

