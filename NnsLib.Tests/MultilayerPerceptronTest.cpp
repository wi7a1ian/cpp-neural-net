#include "pch.h"
#include "TestFixtures.h"

namespace NNSLibTest
{		
	using namespace NNS::Models;
	using namespace NNS::Training;
	using namespace NNS::Activation;

	TEST(MultilayerPerceptronTest, GetOutputActivationForPredefinedWeights)
	{
		// given
		auto network = GetMultilayerPerceptronWithPredefinedWeights();

		// when
		InputLayer input(2); input << 0.0, 1.0;
		network->ComputeOutput(input);

		// then
		EXPECT_EQ((*network).GetActivation(2, 0), network->GetOutputActivation(0));
		EXPECT_TRUE(0.9 < network->GetOutputActivation(0));
	}

	TEST(MultilayerPerceptronTest, GetOutputActivationDerivForPredefinedWeights)
	{
		// given
		auto e = 1'000'000;
		auto network = GetMultilayerPerceptronWithPredefinedWeights();

		// when
		InputLayer input(2); input << 0.0, 1.0;
		network->ComputeOutput(input);

		// then
		EXPECT_EQ(0.923992, trunc(network->GetActivation(2, 0) * e) / e);
		EXPECT_EQ(0.203406, trunc(network->GetActivationDerivative(2, 0) * e) / e);

	}
}