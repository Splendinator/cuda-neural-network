#pragma once

#include <random>
#include <iostream>
#include "kernel.cuh"



//struct Node {
//	
//	float value;	//Activation between 0.0 and 1.0
//	float bias;		//Offset for final value in calculation
//};

class Stage
{
public:
	
	Stage(int nodes);					//Initialises first stage
	Stage(int nodes, Stage *prev);		//Initialises following stages
	~Stage();

	const int numNodes;
	const int numWeights;

	//Node
	float *values;
	float *biases;
	
	//Paths
	float *weights;
	
	Stage *prev;



	void calculate(); //Calculate new values for each node(USING CUDA)
	void randomise(float minWight, float maxWeight, float minBias, float maxBias); //Randomise all weights and bias


};

