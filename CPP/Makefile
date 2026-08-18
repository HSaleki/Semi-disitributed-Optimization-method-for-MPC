# Fallback for machines without cmake. Override EIGEN if it lives elsewhere:
#   make EIGEN=/opt/homebrew/include/eigen3
EIGEN ?= /usr/include/eigen3
CXXFLAGS := -std=c++17 -O2 -Wall -I$(EIGEN)

platoon: platoon.cpp dmpc.hpp
	$(CXX) $(CXXFLAGS) -o $@ platoon.cpp

clean:
	rm -f platoon
