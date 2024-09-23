#pragma once

#include "tensor.h"
#include "backward.h"

namespace tensor {
  namespace backward {

    class Sum : public Backward {
      TNSR backward(TNSR prev) {
        return tensor::ones(tensor::Shape{1}) * prev;
      }
    };

    class NoOp : public Backward {
      TNSR backward(TNSR prev) override {
      }
    };

    template <typename V>
    class Ones : public Backward {
      private:
      V data;
      Shape shape;

      public:
      Ones(V data) : data(data) {
        this->shape = _shape(data);
      }

      TNSR backward(TNSR prev) override {
        return tensor::ones(shape) * prev;
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

    BW noOp() {
      return std::make_shared<NoOp>();
    }

    BW sum() {
      return std::make_shared<Sum>();
    }

    template <typename V>
    BW ones(V data) {
      return std::make_shared<Ones<V>>(data);
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
