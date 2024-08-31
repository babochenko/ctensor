#pragma once

#include "tensor.h"

namespace tensor {
  namespace criterion {
    class CrossEntropyLoss {
      public:
      TNSR X;
      TNSR Y;
      float loss = -1.0;

      float calculate();
      void backward();
    };

    float CrossEntropyLoss::calculate() {
      auto exp = X->exp();
      auto sum = exp->sum();
      exp / sum;

    }
  }
}

