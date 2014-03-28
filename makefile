all:
	g++ -o shudu shudu.cpp sudoku.cpp svm.cpp classes/feature.cpp `pkg-config --cflags --libs opencv`
