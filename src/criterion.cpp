#include "tensor.h"

#include "criterion.h"

namespace tensor {
  namespace criterion {
    TNSR softmax(TNSR X) {
      auto exp = X->exp();
      auto sum = exp->sum();
      auto softmax = exp / sum;

      return softmax;
    }

    TNSR nll(TNSR softmax, TNSR Y) {
      return -((softmax->log() * Y)->sum());
    }

    TNSR CrossEntropyLoss::calculate() {
      return nll(softmax(X), Y);
    }
  }
}

