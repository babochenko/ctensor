#pragma once

#include <string>

class Portfolio {
private:
  int value;
  std::string user_id;

public:
  Portfolio(int value, std::string user_id);
  void calculate();
};

