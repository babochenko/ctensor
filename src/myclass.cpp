#include <iostream>
#include <string>
#include "myclass.h"

Portfolio::Portfolio(int value, std::string user_id) : value(value), user_id(user_id) {
  std::cout << "in constructor" << std::endl;
}

void Portfolio::calculate() {
  std::cout << "goodbye world" << std::endl;
}

