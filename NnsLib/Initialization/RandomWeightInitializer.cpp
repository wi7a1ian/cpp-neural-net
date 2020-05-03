#include "pch.h"
#include "Initialization/RandomWeightInitializer.h"
#include "Common/InterfaceHelpers.h"

namespace NNS
{
	namespace Initialization 
	{

		RandomWeightInitializer::RandomWeightInitializer(WeightUnit magnitude)
			: randomMagnitude(fabs(magnitude))
		{
			// Nop
		}

		void RandomWeightInitializer::Free() const
		{
			delete this;
		}

		void RandomWeightInitializer::InitializeWeights(IFeedforwardNetwork& network)
		{
			const auto networkmap = network.GetNetworkLayerMap();
			network.Rebuild();

			if (auto biasedNetwork = as<Models::IBiased>(network))
			{
				biasedNetwork->SetBiasForAll(1.0);
			}

			std::mt19937 random_generator(static_cast<int>(time(0)));
			std::uniform_real_distribution<WeightUnit> random01(0, 1);

			for (int i = 1; i < networkmap.size(); ++i)  /* For each layer ( minus input layer ). */
			{
				for (int j = 0; j < networkmap[i]; ++j)  /* For each neuron. */
				{
					for (int k = 0; k < networkmap[i - 1]; ++k) /* For each conenction with previous layer. */
					{
						/* It is important to select small initial weights so that all of the units are uncommitted (having activations that are all close to 0.5 - the point of maximal weight change). */
						network.Weight(i, j, k) = randomMagnitude * (1 - 2 * random01(random_generator)); /* Generate random number from range <-x; x> , best is <-0.5; 0.5> */
					}
				} 
			}
		}
	}
}