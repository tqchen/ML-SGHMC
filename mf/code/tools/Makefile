export CC  = gcc
export CXX = g++
export CFLAGS = -Wall -O3 -msse2 -I..

# specify tensor path
BIN = make_feature_buffer line_shuffle 
OBJ = apex_svd_data.o
.PHONY: clean all

all: $(BIN)
export LDFLAGS= -pthread -lm 

apex_svd_data.o:../apex_svd_data.cpp ../apex_svd_data.h
make_feature_buffer:make_feature_buffer.cpp apex_svd_data.o ../apex_svd_data.h
line_shuffle:line_shuffle.cpp 

$(BIN) : 
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(filter %.cpp %.c, $^)

install:
	cp -f -r $(BIN)  $(INSTALL_PATH)

clean:
	$(RM) $(OBJ) $(BIN) *~

