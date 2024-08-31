#pragma once

#include "tensor.h"

namespace tensor {
  namespace criterion {
    class CrossEntropyLoss {
      public:
      TNSR X;
      TNSR Y;
      float loss = -1.0;

      CrossEntropyLoss(TNSR X, TNSR Y) : X(X), Y(Y) {
        compare_shapes(X->shape, Y->shape);
      }
      
      float calculate();
      void backward();
    };

    float CrossEntropyLoss::calculate() {
      auto exp = X->exp();
      auto sum = exp->sum();
      auto softmax = exp / sum;

      
    }
  }
}

