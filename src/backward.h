#pragma once

#include "tensor.h"

namespace tensor {
  namespace backward {

    class Sum : public Backward {
      TNSR backward(TNSR prev) {
        return tensor::ones(tensor::Shape{1}) * prev;
      }
    };

    class NoOpV : public Backward {
      private:
      V_VEC data;

      public:
      NoOpV(V_VEC data) : data(data) {}

      TNSR backward(TNSR prev) override {
        return tensor::ones(Shape(1, data.size())) * prev;
      }
    };

    class NoOp : public Backward {
      private:
      P_VEC data;

      public:
      NoOp(P_VEC data) : data(data) {}

      TNSR backward(TNSR prev) override {
        return tensor::ones(_shape(data)) * prev;
      }
    };

    class NoOpP : public Backward {
      private:
      TNSR data;

      public:
      NoOpP(TNSR data) : data(data) {}

      TNSR backward(TNSR prev) override {
        return tensor::ones(data->shape) * prev;
      }
    };

   class Mul : public Backward {
      private:
      TNSR data;

      public:
      Mul(TNSR data) : data(data) {}

      TNSR backward(TNSR prev) override {
        return data * prev;
      }
    };

    class MatMulLeft : public Backward {
      private:
      TNSR other;

      public:
      MatMulLeft(TNSR other) : other(other) {}

      TNSR backward(TNSR prev) override {
        return prev->mul(other->T());
      }
    };

    class MatMulRight : public Backward {
      private:
      TNSR other;

      public:
      MatMulRight(TNSR other) : other(other) {}

      TNSR backward(TNSR prev) override {
        return other->T()->mul(prev);
      }
    };

    class Pow : public Backward {
      private:
      TNSR data;
      int power;

      public:
      Pow(TNSR data, int power) : data(data), power(power) {}

      TNSR backward(TNSR prev) override {
        int newPow = power - 1;
        auto res = data->pow(power - 1);
        if (newPow < 0) {
          res = -res;
        }
        return res;
      }
    };

    BW sum() {
      return std::make_shared<Sum>();
    }

    BW noOpV(V_VEC data) {
      return std::make_shared<NoOpV>(data);
    }

    BW noOp(P_VEC data) {
      return std::make_shared<NoOp>(data);
    }

    BW noOpP(TNSR data) {
      return std::make_shared<NoOpP>(data);
    }

    BW mul(TNSR data) {
      return std::make_shared<Mul>(data);
    }

    BW matMulLeft(TNSR data) {
      return std::make_shared<MatMulLeft>(data);
    }

    BW matMulRight(TNSR data) {
      return std::make_shared<MatMulRight>(data);
    }

  }
}
