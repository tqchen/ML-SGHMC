export CC  = gcc
export CXX = g++
export CFLAGS = -Wall -O3 -msse2 -fopenmp -I..

# specify tensor path
BIN = bpmf_hmc
OBJ = bpmf_hmc.o apex_svd_data.o
.PHONY: clean all

all: $(BIN) $(OBJ)
export LDFLAGS= -pthread -lm 

bpmf_hmc: svd_feature.cpp $(OBJ) apex_svd_data.h 
bpmf_hmc.o: apex_svd.cpp hmc-bpmf/*.h
apex_svd_data.o: apex_svd_data.cpp apex_svd_data.h

$(BIN) : 
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

install:
	cp -f -r $(BIN)  $(INSTALL_PATH)

clean:
	$(RM) $(OBJ) $(BIN) *~
