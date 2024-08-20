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

  class Tensor {
    using Vec = std::variant<V_VEC, P_VEC>;

    public:
    Vec vector;
    Shape shape;

    Tensor(V_VEC &v, Shape s) : shape(s) , vector(v) {}
    Tensor(P_VEC v, Shape s) : shape(s), vector(v) {}

    TNSR flatten();
    TNSR T();
    TNSR T(int dim1, int dim2);
    TNSR mul(TNSR other);

    virtual std::string _str(int depth);
    virtual std::string str();
    void print(std::ostream &os) { os << str(); }

    friend std::ostream& operator<<(std::ostream &os, Tensor &t) {
      t.print(os);
      return os;
    }
  };

  TNSR arange(int start, int endExclusive);
  TNSR zeros(Shape &shape);
  TNSR ones(Shape &shape);

  namespace random {
    TNSR uniform(float min, float max, Shape &shape);
    TNSR uniform(Shape &shape);
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

