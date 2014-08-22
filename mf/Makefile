
# you will need to change path of data to path of your ml-1m data
export DATA=~/data/recom/ml-1m/ratings.dat

.PHONY: all clean

all: fold$(k).buffer
clean:
	rm -rf *.txt *.buffer *.group *~ */*~

data.txt: $(DATA)
	python	python/mkdata.py $+ $@ > info.txt

fold$(k).txt: data.txt
	python	python/mknfold.py $+ $(k)

fold$(k).group: fold$(k).txt
	code/tools/line_shuffle $+ $@ 0

fold$(k).buffer: fold$(k).group
	code/tools/make_feature_buffer $+ $@
