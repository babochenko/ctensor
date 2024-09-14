#include <gtest/gtest.h>
#include <string>

#include "tensor.h"
#include "tensor.cpp"

#include "criterion.h"
#include "criterion.cpp"

void test_arange(int from, int toExclusive, std::string expected) {
  std::stringstream ss;
  auto t = tensor::arange(from, toExclusive);
  ss << *t;

  EXPECT_EQ(ss.str(), expected);
}

TEST(arange, 0arg) {
  test_arange(0, 0, "[]");
}

TEST(arange, 1arg) {
  test_arange(0, 1, "[0]");
}

TEST(arange, Nargs) {
  test_arange(3, 7, "[3,4,5,6]");
}

void expect(tensor::TNSR tensor, std::string expected) {
  std::stringstream ss;
  ss << *tensor;

  EXPECT_EQ(ss.str(), expected);
}

void expect(float actual, float expected) {
  EXPECT_EQ(actual, expected);
}

TEST(Tensor, ones) {
  auto t = tensor::ones(tensor::Shape{3,4,3});

  expect(t, ""
    "[[[1,1,1],\n"
    "  [1,1,1],\n"
    "  [1,1,1],\n"
    "  [1,1,1]],\n"
    " [[1,1,1],\n"
    "  [1,1,1],\n"
    "  [1,1,1],\n"
    "  [1,1,1]],\n"
    " [[1,1,1],\n"
    "  [1,1,1],\n"
    "  [1,1,1],\n"
    "  [1,1,1]]]");
}

TEST(Tensor, negate1) {
  auto t1 = tensor::arange(0, 5);
  auto res = -t1;
  expect(res, "[-0,-1,-2,-3,-4]");
}

TEST(Tensor, negate2) {
  auto t1 = tensor::arange(0, 4)->resize(tensor::Shape{2,2});
  auto res = -t1;
  expect(res, ""
    "[[-0,-1],\n"
    " [-2,-3]]");
}

TEST(Tensor, exp1) {
  auto t1 = tensor::arange(0, 3);
  auto res = t1->exp();
  expect(res, "[1,2.71828,7.38906]");
}

TEST(Tensor, exp2) {
  auto t1 = tensor::arange(0, 4)->resize(tensor::Shape{2,2});
  auto res = t1->exp();
  expect(res, ""
    "[[1,2.71828],\n"
    " [7.38906,20.0855]]");
}

TEST(Tensor, log1) {
  auto t1 = tensor::arange(0, 3);
  auto res = t1->log();
  expect(res, "[-inf,0,0.693147]");
}

TEST(Tensor, log2) {
  auto t1 = tensor::arange(0, 4)->resize(tensor::Shape{2,2});
  auto res = t1->log();
  expect(res, ""
    "[[-inf,0],\n"
    " [0.693147,1.09861]]");
}

TEST(Tensor, log3) {
  auto _x = std::vector<float>{1.0, 0.00001, 0.00001};
  auto _y = std::vector<float>{1.0, 0.0, 0.0};
  auto X = tensor::tnsr(_x);
  auto Y = tensor::tnsr(_y);

  auto log = X->log();
  expect(log, "[0,-11.5129,-11.5129]");

  auto mul = log * Y;
  expect(mul, "[0,-0,-0]");
}

TEST(Tensor, add1) {
  auto t1 = tensor::arange(0, 5);
  auto t2 = tensor::arange(0, 5);
  auto add = t1 + t2;
  expect(add, "[0,2,4,6,8]");
}

TEST(Tensor, add2) {
  auto t1 = tensor::ones(tensor::Shape{2,2});
  auto t2 = tensor::ones(tensor::Shape{2,2});
  auto add = t1 + t2;
  expect(add, ""
    "[[2,2],\n"
    " [2,2]]");
}

TEST(Tensor, mul0) {
  auto t1 = tensor::ones(tensor::Shape{2,2});
  auto mul = 0.5 * t1;
  expect(mul, ""
    "[[0.5,0.5],\n"
    " [0.5,0.5]]");
}

TEST(Tensor, mul1) {
  auto t1 = tensor::ones(tensor::Shape{2,2});
  auto mul = tensor::tnsr(tensor::V_VEC{0.5}) * t1;
  expect(mul, ""
    "[[0.5,0.5],\n"
    " [0.5,0.5]]");
}

TEST(Tensor, mul2) {
  auto t1 = tensor::arange(0, 4, tensor::Shape{2,2});
  auto t2 = tensor::arange(0, 4, tensor::Shape{2,2});
  auto mul = t1 * t2;
  expect(mul, ""
    "[[0,1],\n"
    " [4,9]]");
}

TEST(Tensor, mul3) {
  auto t1 = tensor::arange(0, 4, tensor::Shape{2,2});
  auto t2 = tensor::arange(0, 4, tensor::Shape{2,2});
  auto mul = t1->mul(t2);
  expect(mul, ""
    "[[2,3],\n"
    " [6,11]]");
}

TEST(Tensor, div1) {
  auto t1 = tensor::ones(tensor::Shape{2,2});
  auto mul = t1 / 2;
  expect(mul, ""
    "[[0.5,0.5],\n"
    " [0.5,0.5]]");
}

TEST(Tensor, div2) {
  auto t1 = tensor::ones(tensor::Shape{2,2});
  auto mul = t1 / tensor::arange(2,3);
  expect(mul, ""
    "[[0.5,0.5],\n"
    " [0.5,0.5]]");
}

TEST(Tensor, resize1) {
  auto t = tensor::arange(0, 4);
  auto r = t->resize(tensor::Shape{2,2});
  expect(r, ""
    "[[0,1],\n"
    " [2,3]]");
}

TEST(Tensor, resize2) {
  auto t = tensor::arange(0, 6, tensor::Shape{2,3});
  expect(t, ""
    "[[0,1,2],\n"
    " [3,4,5]]");
}

TEST(Tensor, resize3) {
  auto t = tensor::arange(0, 12, tensor::Shape{2,2,3});
  expect(t, ""
    "[[[0,1,2],\n"
    "  [3,4,5]],\n"
    " [[6,7,8],\n"
    "  [9,10,11]]]");
}

TEST(Tensor, sum1) {
  auto t = tensor::arange(0, 1);
  auto sum = t->sum()->item();
  expect(sum, 0.0);
}

TEST(Tensor, sum2) {
  auto t = tensor::arange(0, 4);
  auto sum = t->sum()->item();
  expect(sum, 6.0);
}

TEST(Tensor, sum3) {
  auto t = tensor::arange(0, 8, tensor::Shape{2,2,2});
  auto sum = t->sum()->item();
  expect(sum, 28.0);
}

TEST(Tensor, dot0) {
  auto t1 = tensor::tnsr(std::vector<float>{1.0});
  auto t2 = tensor::tnsr(std::vector<float>{1.0});
  expect(t1->dot(t2), 1.0f);
}

TEST(Tensor, dot1) {
  auto t1 = tensor::tnsr(std::vector<float>{1.0, 1.0, 1.0});
  auto t2 = tensor::tnsr(std::vector<float>{1.0, 1.0, 1.0});
  expect(t1->dot(t2), 3.0f);
}

TEST(Tensor, dot2) {
  auto t1 = tensor::tnsr(std::vector<float>{1.0, 2.0, 1.0});
  auto t2 = tensor::tnsr(std::vector<float>{2.0, 3.0, 1.0});
  expect(t1->dot(t2), 9.0f);
}

TEST(Criterion, softmax0) {
  auto t = tensor::arange(0, 1);
  auto softmax = tensor::criterion::softmax(t);
  expect(softmax, "[1]");
}

TEST(Criterion, softmax1) {
  auto t = tensor::arange(0, 4);
  auto softmax = tensor::criterion::softmax(t);
  expect(softmax, "[0.0320586,0.0871443,0.236883,0.643914]");
}

TEST(Criterion, softmax2) {
  auto t = tensor::arange(0, 4, tensor::Shape{2,2});
  auto softmax = tensor::criterion::softmax(t);
  expect(softmax, ""
    "[[0.0320586,0.0871443],\n"
    " [0.236883,0.643914]]");
}

TEST(Criterion, softmax3) {
  auto _x = std::vector<float>{1.0, 0.00001, 0.00001};
  auto X = tensor::tnsr(_x);
  auto softmax = tensor::criterion::softmax(X);
  expect(softmax, "[0.576114,0.211943,0.211943]");
}

TEST(Criterion, nll1) {
  auto _x = std::vector<float>{1.0, 0.00001, 0.00001};
  auto _y = std::vector<float>{1.0, 0.0, 0.0};

  auto X = tensor::tnsr(_x);
  auto Y = tensor::tnsr(_y);
  auto loss = tensor::criterion::nll(X, Y)->item();
  expect(loss, 0.0);
}

TEST(Criterion, nll2) {
  auto _x = std::vector<float>{0.576114, 0.211943, 0.211943};
  auto _y = std::vector<float>{1.0, 0.0, 0.0};

  auto X = tensor::tnsr(_x);
  auto Y = tensor::tnsr(_y);
  auto loss = tensor::criterion::nll(X, Y)->item();
  expect(loss, 0.551449716);
}

TEST(Criterion, crossent1) {
  auto _x = std::vector<float>{10.0, -10.0, -10.0};
  auto _y = std::vector<float>{1.0, 0.0, 0.0};

  auto X = tensor::tnsr(_x);
  auto Y = tensor::tnsr(_y);
  auto loss = tensor::criterion::CrossEntropyLoss(X, Y);
  expect(loss.calculate()->item(), 0.0);
}

TEST(Criterion, crossent2) {
  auto _x = std::vector<float>{1.0, 0.0, 0.0};
  auto _y = std::vector<float>{1.0, 0.0, 0.0};

  auto X = tensor::tnsr(_x);
  auto Y = tensor::tnsr(_y);
  auto loss = tensor::criterion::CrossEntropyLoss(X, Y);
  expect(loss.calculate()->item(), 0.551444769);
}

tensor::TNSR _X() {
  return tensor::tnsr(std::vector<float>{
    0.8413, 0.8408, 0.1184,
    0.8690, 0.0695, 0.4731,
    0.0101, 0.9498, 0.3768,
  })->resize(tensor::Shape{3,3});
}

tensor::TNSR _W() {
  return tensor::tnsr(std::vector<float>{
    0.2939, 0.3367, 0.1780,
  })->resize(tensor::Shape{3,1});
}

TEST(Backprop, matmul) {
  auto X = _X();
  auto W = _W();

  auto logs = X->mul(W)->resize(tensor::Shape{3});
  expect(logs, "[0.551431,0.363012,0.389836]");

  auto y = tensor::tnsr(tensor::V_VEC{1.0, 0.0, 0.0});
  auto loss = tensor::criterion::CrossEntropyLoss(logs, y);
  auto _loss = loss.calculate();
  expect(_loss->item(), 0.985463798);

  _loss->backward();
  expect(_loss->grad, "[1]");

  expect(X->grad, ""
       "[[-0.1842,-0.2110,-0.1115],\n"
       " [0.0909,0.1041,0.0550],\n"
       " [0.0933,0.1069,0.0565]]");

  expect(W->grad, ""
       "[[-0.2554],\n"
       " [-0.2038],\n"
       " [ 0.1917]]");
}

TEST(Backprop, base_tensor) {
  expect(_X(), ""
    "[[0.8413,0.8408,0.1184],\n"
    " [0.869,0.0695,0.4731],\n"
    " [0.0101,0.9498,0.3768]]");
}

TEST(Backprop, sum) {
  auto t = _X();
  auto sum = t->sum();

  sum->backward();

  expect(sum->grad, "[1]");
  expect(t->grad, ""
    "[[1,1,1],\n"
    " [1,1,1],\n"
    " [1,1,1]]");
}

