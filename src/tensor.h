#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <memory>

namespace tensor {

  class Tensor;

  using Shape = std::vector<int>;
  using TNSR = std::shared_ptr<Tensor>;
  using P_VEC = std::vector<TNSR>;
  using V_VEC = std::vector<float>;

  class Tensor : public std::enable_shared_from_this<Tensor> {
    using Vec = std::variant<V_VEC, P_VEC>;

    public:
    Vec vector;
    Shape shape;

    Tensor(V_VEC &v, Shape s) : shape(s) , vector(v) {}
    Tensor(P_VEC v, Shape s) : shape(s), vector(v) {}

    TNSR resize(const Shape &shape);
    TNSR flatten();
    TNSR T();
    TNSR T(int dim1, int dim2);
    TNSR mul(TNSR other);

    TNSR exp();
    double sum();

    virtual std::string _str(int depth);
    virtual std::string str();
    void print(std::ostream &os) { os << str(); }

    friend std::ostream& operator<<(std::ostream &os, Tensor &t) {
      t.print(os);
      return os;
    }
  };

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
}

