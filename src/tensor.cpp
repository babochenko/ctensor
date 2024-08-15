#include <string>
#include <sstream>

#include "tensor.h"

namespace tensor {

  Tensor::Tensor(int start, int end) : start(start), end(end) {
  }

  std::string Tensor::toString() {
    std::stringstream ss;
    ss << "Tensor(" << start << ", " << end << ")";
    return ss.str();
  } 

  tensor::Tensor arange(int start, int endExclusive) {
    Tensor t(start, endExclusive);
    return t;
  }
}

