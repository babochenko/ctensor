all:
	make clean && make build && make test

build:
	mkdir build && cd build && cmake .. && make

clean:
	rm -rf build

test:
	g++ -std=c++17 -I/opt/homebrew/opt/googletest/include/gtest -L/opt/homebrew/opt/googletest/lib src/tensor_test.cpp -lgtest -lgtest_main -pthread -o build/tensor_test && build/tensor_test

run:
	./build/ctensor

