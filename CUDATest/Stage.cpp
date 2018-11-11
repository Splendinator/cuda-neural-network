#include "Stage.h"


Stage::Stage(int nodes, Stage *prev) : numNodes(nodes), numWeights(numNodes * prev->numNodes)
{	
	this->values = new float[nodes];
	this->biases = new float[nodes];
	this->prev = prev;
	weights = new float[numWeights];
}

Stage::Stage(int nodes) : numNodes(nodes), numWeights(0)
{
	this->values = new float[nodes];
	this->biases = new float[nodes];;
	this->prev = nullptr;
}



Stage::~Stage()
{
	delete[] values;
	delete[] biases;

	if (weights) {
		delete[] weights;
	}
}

void Stage::calculate()
{
	
	//Multiply
	float *dp, *dvalues, *dbiases;

	cudaMalloc(&dp, sizeof(float) * (numWeights + prev->numNodes + numNodes));
	cudaMemcpy(dp, weights, sizeof(float) * numWeights, cudaMemcpyHostToDevice);
	
	dvalues = dp + numWeights;
	dbiases = dvalues + numNodes;
	
	cudaMemcpy(dvalues, prev->values, sizeof(float) * prev->numNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(dbiases, biases, sizeof(float) * numNodes, cudaMemcpyHostToDevice);

	multNodes <<<1, numWeights>>> (dp, dvalues, prev->numNodes);	//Multiply the previous stage's node values with the path weights
	

	total <<<numNodes,1>>> (dp, prev->numNodes, dbiases);					//Get the total weight of each node by totalling all the calculated weight*prevValue

	//Copy newly calculated total node weights back onto host.
	for (int i = 0; i < numNodes; ++i) {
		cudaMemcpy(values + i, dp+i*prev->numNodes, sizeof(float), cudaMemcpyDeviceToHost);
	}
	//total <<<1, 1 >>>(dp, 9);

	//cudaMemcpy(weights, dp, sizeof(float) * 9, cudaMemcpyDeviceToHost);
}


void Stage::randomise(float minWeight, float maxWeight, float minBias, float maxBias)
{

	static const int BLOCK_SIZE = 256;

	float *dp;

	int blocks = (numWeights + BLOCK_SIZE-1) / BLOCK_SIZE;

	cudaMalloc(&dp, sizeof(float) * (blocks*BLOCK_SIZE + 2));

	cudaMemcpy(dp, &minWeight, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dp + 1, &maxWeight, sizeof(float), cudaMemcpyHostToDevice);

	randf <<<blocks, BLOCK_SIZE>>>(dp + 2, dp, dp + 1);

	cudaMemcpy(weights, dp + 2, sizeof(float) * numWeights, cudaMemcpyDeviceToHost);

	cudaFree(dp);

}


