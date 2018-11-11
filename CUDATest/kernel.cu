#include "kernel.cuh"






__global__ void randf(float *f, float *min, float *max) {

	int seed1 = (blockIdx.x * blockDim.x + threadIdx.x) & 255;
	int seed2 = ((blockIdx.x * blockDim.x + threadIdx.x) / 256) & 255;

	static const int p[256] =    //Array has loop property. Can keep calling p[p[x]] and loop through all numbers
	{   
		58, 245, 227, 194, 59, 228, 189, 88, 218, 140, 19, 93, 222, 186, 252, 27, 175, 73, 212, 24, 56, 209, 210,
		149, 83, 200, 214, 8, 160, 197, 195, 36, 98, 39, 112, 202, 159, 165, 12, 152, 179, 44, 64, 139, 68, 23,
		239, 91, 102, 211, 226, 141, 60, 190, 72, 111, 45, 134, 171, 30, 231, 48, 104, 246, 78, 114, 106, 207,
		31, 131, 37, 237, 113, 176, 230, 242, 136, 11, 35, 101, 81, 133, 79, 84, 53, 251, 151, 80, 0, 18, 107,
		28, 74, 183, 50, 77, 85, 155, 21, 250, 17, 95, 99, 192, 198, 2, 243, 96, 34, 43, 9, 204, 14, 52, 143,
		70, 6, 42, 205, 236, 148, 232, 201, 213, 49, 108, 10, 138, 92, 119, 193, 4, 105, 125, 181, 219, 216, 51,
		110, 166, 62, 233, 241, 224, 22, 29, 130, 255, 146, 199, 103, 217, 234, 156, 145, 32, 38, 126, 75, 158, 254,
		223, 46, 3, 40, 184, 25, 13, 82, 185, 54, 135, 144, 124, 154, 7, 47, 132, 100, 15, 67, 66, 157, 87, 121, 76,
		180, 220, 20, 203, 229, 127, 225, 109, 33, 122, 215, 235, 150, 97, 94, 89, 168, 120, 196, 206, 164, 153, 247,
		1, 117, 253, 86, 208, 191, 163, 178, 5, 129, 118, 55, 182, 128, 249, 167, 115, 169, 57, 174, 187, 16, 161, 90,
		147, 162, 240, 188, 173, 172, 170, 61, 116, 177, 142, 41, 71, 137, 248, 63, 244, 123, 65, 221, 238, 69, 26 
	};
	
	
	f[blockIdx.x * blockDim.x + threadIdx.x] = (*max - *min) * ((((p[(p[seed1] + seed2)&255] + p[seed2])&255) | (p[(p[seed2] + seed1)&255]) << 8) / 65535.0f) + *min;
	
}


__global__ void add(float *a, float *b) {
	a[blockIdx.x * blockDim.x + threadIdx.x] += b[blockIdx.x * blockDim.x + threadIdx.x];
}

__global__ void total(float *f, int size, float *bias) {

	float *p = f + size*(blockIdx.x * blockDim.x + threadIdx.x);

	while (size > 1) {
		
		if ((size % 2)) {
			p[size - 2] += p[size - 1];
			--size;
			
		}
		size /= 2;
		add <<<1, size>>> (p, p + size);
		
		cudaDeviceSynchronize();	//Don't start the next iteration until this one has finished since they depend on each other.

		
	}
	//Sigmoid
	*p = (1 / (1 + powf(E, - (bias[(blockIdx.x * blockDim.x + threadIdx.x)] + *p))));
}

__global__ void mult(float *f, float *m) {
	f[blockIdx.x * blockDim.x + threadIdx.x] *= m[blockIdx.x * blockDim.x + threadIdx.x];
}

__global__ void multNodes(float *weights, float *values, int numNodes) {
	weights[blockIdx.x * blockDim.x + threadIdx.x] *= values[(blockIdx.x * blockDim.x + threadIdx.x) % numNodes] ;
}



