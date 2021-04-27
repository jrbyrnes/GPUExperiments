#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <math.h>

#define NTHREAD 16
#define MAXLENGTH 131072

void saxpy(float *x, float *y, int start, int n, float a, int nIterations) {
	for (int i = 0; i < nIterations; i++) {
		for (int i = start; i < start + n; i++) {
			y[i] = a * x[i] + y[i];
		}
	}
}

void fillArrays(float *x, float *y, int size) {
	for (int i = 0; i < size; i++) {
		x[i] = i;
		y[i] = size - 1 - i;
	}
}


void printArray(const std::string arrayName,
                 const float * arrayData,
                 const unsigned int length)
{
  int numElementsToPrint = (256 < length) ? 256 : length;
  std::cout << std::endl << arrayName << ":" << std::endl;
  for(int i = 0; i < numElementsToPrint; ++i)
    std::cout << arrayData[i] << " ";
  std::cout << std::endl;
}



int main() {
	unsigned long long LENGTH = (unsigned long long)2147483648 * (unsigned long long)512;
	std::vector<std::thread> ThreadManager(NTHREAD);
	int workUnitSize, nIterations, arraySize;
	if (LENGTH > MAXLENGTH) {
		workUnitSize = MAXLENGTH/NTHREAD;
		nIterations = LENGTH/MAXLENGTH;
		arraySize = MAXLENGTH;
	}
	else {
		workUnitSize = LENGTH/NTHREAD;
		nIterations = 1;
		arraySize = LENGTH;
	}

	float *x = new float[arraySize];
	float *y = new float[arraySize];

	fillArrays(x,y, arraySize);



	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();


	for (int i = 0; i < NTHREAD; i++) {
		ThreadManager[i] = std::thread(saxpy, x, y, workUnitSize * i, workUnitSize, 2, nIterations);
	}

	for (int i = 0; i < NTHREAD; i++) {
		ThreadManager[i].join();
	}

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	printArray("Y", y, arraySize);

	

	std::cout << "Elapsed Time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
	printf("Each thread performed %d iterations of %d operations, for a total of %lld operations (2 ^ %d)\n",nIterations, workUnitSize, LENGTH, (int)log2(LENGTH));
 
}
