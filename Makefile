
CXX = g++
CXXFLAGS = -std=c++17 -Wall -g


INCLUDES = -I./cnpy -I./include

LIBS = -lz -pthread

SRCS = main.cpp cnpy/cnpy.cpp include/MurmurHash3.cpp

TARGET = main

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(SRCS) $(LIBS)

clean:
	rm -f $(TARGET)
	rm -f ./tmp/*

run: $(TARGET)
	./$(TARGET) -m ZRingDME -k 1

.PHONY: all clean run
