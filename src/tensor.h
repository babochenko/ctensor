#pragma once

#include <string>

namespace tensor {

  class Tensor {
    private:
    int start;
    int end;

    public:
    Tensor(int start, int end);
    std::string toString();
  };

  tensor::Tensor arange(int start, int endExclusive);

}

