build:
	mkdir build && cd build && cmake .. && make

clean:
	rm -rf build

run:
	./build/ctensor

all: build

