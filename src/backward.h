#pragma once

#include "tensor.h"
#include "backward.h"

namespace tensor {
  namespace backward {

    BW noOp() {
      return [](TNSR prev) { return prev; };
    }

    BW sum() {
      return [](TNSR prev) {
        return tensor::ones(tensor::Shape{1}) * prev;
      };
    }

    template <typename X>
    BW ones(X x) {
      Shape s = _shape(x);
      return [&](TNSR prev) {
        return tensor::ones(s) * prev;
      };
    }

    BW mul(TNSR x) {
      return [&](TNSR prev) { return x * prev; };
    }

    BW log(TNSR x) {
      return [&](TNSR prev) { return 1 / x * prev; };
    }

    BW exp(TNSR exp) {
      return [&](TNSR prev) { return exp * prev; };
    }

    BW pow(TNSR x, int power) {
      return [&](TNSR prev) {
        int newPow = power - 1;
        auto res = x->pow(power - 1);
        if (newPow < 0) {
          res = -res;
        }
        return res;
      };
    }

    BW matMulLeft(TNSR other) {
      return [&](TNSR prev) {
        return prev->mul(other->T());
      };
    }

    BW matMulRight(TNSR other) {
      return [&](TNSR prev) {
        return other->T()->mul(prev);
      };
    }

  }
}
