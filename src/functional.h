#pragma once

#include "tensor.h"
#include "backward.h"

namespace func {

  using tensor::PARENTS;

  template <typename T1, typename T2>
  constexpr bool is_() {
    using V = std::decay_t<T1>;
    return std::is_same_v<V, T2>;
  } 

  template <typename Op>
  tensor::TNSR _map(tensor::V_VEC data, Op op) {
    tensor::V_VEC out{};

    for (size_t i = 0; i < data.size(); i++) {
      auto result = op(data[i]);
      out.push_back(result);
    }

    return tensor::tnsr(out);
  };

  template <typename Op>
  float _visit(tensor::V_VEC data, Op op) {
    for (size_t i = 0; i < data.size(); i++) {
      op(data[i]);
    }
    return 0.0f;
  };

  template <typename Op>
  tensor::TNSR map(tensor::TNSR tensor, float val, Op operation) {
    auto op_float = [&](float t) { return operation(t, val); };
    return map(tensor, op_float);
  }

  template <typename OP>
  tensor::TNSR map(tensor::TNSR tensor, OP operation) {
    auto dim = tensor->shape[0];

    return std::visit([&](auto&& vector) {
      using ELMNT = decltype(vector[0]);
      if constexpr (is_<ELMNT, tensor::TNSR>()) {
        tensor::P_VEC res{};
        for (auto t: vector) {
          res.push_back(map(t, operation));
        }
        return tensor::tnsr(res);
      } else if constexpr (is_<ELMNT, float>()) {
        return _map(vector, operation);
      } else {
        return tensor::tnsr(tensor::V_VEC{});
      }
    }, tensor->vector);
  }

  template <typename Op>
  void visit(tensor::TNSR tensor, float val, Op operation) {
    auto v = [&](float t) { return operation(t, val); };
    visit(tensor, v);
  }

  template <typename Op>
  void visit(tensor::TNSR tensor, Op operation) {
    auto dim = tensor->shape[0];

    tensor->visit(
      [&](tensor::P_VEC vec) { 
        for (auto el: vec) {
          visit(el, operation);
        }
      },
      [&](tensor::V_VEC vec) { _visit(vec, operation); }
    );
  }

  template <typename Op>
  tensor::TNSR merge(
    tensor::TNSR &tensor1,
    tensor::TNSR &tensor2,
    Op op,
    tensor::BW t1bw,
    tensor::BW t2bw
  ) {
    if (tensor2->shape == tensor::Shape{1}) {
      return func::map(tensor1, tensor2->item(), op);
    }

    tensor::compare(tensor1->shape, tensor2->shape, "merge() shapes");

    tensor1->set_backward(t1bw);
    tensor2->set_backward(t2bw);
    PARENTS prnts{tensor1, tensor2};
    auto noop = tensor::backward::noOp();

    return std::visit([&](auto&& t1, auto&& t2) {
      if constexpr (is_<decltype(t1[0]), tensor::TNSR>() && is_<decltype(t2[0]), tensor::TNSR>()) {
        tensor::P_VEC data{};
        for (size_t i = 0; i < t1.size(); i++) {
          data.push_back(merge(t1[i], t2[i], op, noop, noop));
        }
        return tensor::tnsr(data, tensor::backward::ones(data), prnts);

      } else if constexpr (is_<decltype(t1[0]), float>() && is_<decltype(t2[0]), float>()) {
        tensor::V_VEC data{};

        for (size_t i = 0; i < t1.size(); i++) {
          auto res = op(t1[i], t2[i]);
          data.push_back(res);
        }

        return tensor::tnsr(data);
      } else {

        tensor::V_VEC data{};
        return tensor::tnsr(data);
      }
    }, tensor1->vector, tensor2->vector);
  }
}

