#pragma once

#include "tensor.h"

namespace tensor {
  namespace criterion {
    class CrossEntropyLoss {
      public:
      TNSR X;
      TNSR Y;
      double loss = -1.0;

      double calculate();
      void backward();
    };

    double CrossEntropyLoss::calculate() {

    }
  }
}

