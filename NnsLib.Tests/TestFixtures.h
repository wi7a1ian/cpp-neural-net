#pragma once

#include <Models/MultilayerPerceptron.h>

namespace NNSLibTest
{
	using namespace NNS::Models;

	std::unique_ptr<MultilayerPerceptron> GetMultilayerPerceptronWithPredefinedWeights();

	// TODO: GTest test fixtures here (parametrized also)
}