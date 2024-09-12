#pragma once

#include "tensor.h"

namespace tensor {
  namespace backward {

    class Sum : public Backward {
      TNSR backward() {
        return tensor::ones(tensor::Shape{1});
      }
    };

    class NoOpV : public Backward {
      private:
      V_VEC data;

      public:
      NoOpV(V_VEC data) : data(data) {}

      TNSR backward() override {
        return tensor::ones(Shape(1, data.size()));
      }
    };

    class NoOp : public Backward {
      private:
      P_VEC data;

      public:
      NoOp(P_VEC data) : data(data) {}

      TNSR backward() override {
        return tensor::ones(_shape(data));
      }
    };

  }
}
