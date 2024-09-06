#include "tensor.h"

#include "criterion.h"

namespace tensor {
  namespace criterion {
    TNSR softmax(TNSR X) {
      auto exp = X->exp();
      auto sum = exp->sum()->item();
      auto softmax = exp / sum;

      return softmax;
    }

    float nll(TNSR softmax, TNSR Y) {
      return -((softmax->log() * Y)->sum())->item();
    }

    float CrossEntropyLoss::calculate() {
      return nll(softmax(X), Y);
    }
  }
}

