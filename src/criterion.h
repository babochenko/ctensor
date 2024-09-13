#pragma once

#include "tensor.h"

namespace tensor {
  namespace criterion {

    TNSR softmax(TNSR X);
    TNSR nll(TNSR softmax, TNSR Y);

    class CrossEntropyLoss {
      public:
      TNSR X;
      TNSR Y;
      float loss = -1.0;

      CrossEntropyLoss(TNSR X, TNSR Y) : X(X), Y(Y) {
        compare_shapes(X->shape, Y->shape);
      }

      TNSR calculate();
      void backward();
    };
  }
}

