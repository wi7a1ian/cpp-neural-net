#include "pch.h"
#include "TestFixtures.h"

namespace NNSLibTest
{
	std::unique_ptr<MultilayerPerceptron> GetMultilayerPerceptronWithPredefinedWeights()
	{
		std::unique_ptr<MultilayerPerceptron> network(new MultilayerPerceptron({ 2, 2, 1 }));

		// SetWeights
		network->Weight(1, 0, 0) = -4.8f;
		network->Weight(1, 0, 1) = 4.6f;
		network->Weight(1, 1, 0) = 5.1f;
		network->Weight(1, 1, 1) = -5.2f;

		network->Weight(2, 0, 0) = 5.9f;
		network->Weight(2, 0, 1) = 5.2f;

		// SetBias
		network->Bias(1, 0) = -2.6f;
		network->Bias(1, 1) = -3.2f;
		network->Bias(2, 0) = -2.7f;

		return network;
	}
}