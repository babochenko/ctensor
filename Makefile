build:
	mkdir build && cd build && cmake .. && make

clean:
	rm -rf build

all: build

