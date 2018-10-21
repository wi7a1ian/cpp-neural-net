#include "pch.h"

namespace NNSLibTest
{		
	using namespace NNS::Models;
	using namespace NNS::Training;
	using std::unique_ptr;

	static const int numberOfTests = 1000000;
	static const int numberOfCompetetiveNeurons = 100;

	TEST(DISABLED_KohonenNetworkTest, PredefinedWeights)
	{
		auto network = std::make_unique<KohonenNetwork>(5, numberOfCompetetiveNeurons);
		InputLayer input; input << 1, 2, 3, 4, 5;

		for (int i = 0; i < numberOfTests; ++i)
		{
			network->ComputeOutput(input);
		}
	}
}