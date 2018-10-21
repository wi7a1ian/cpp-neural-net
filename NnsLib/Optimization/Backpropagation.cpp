#include "pch.h"
#include "Common/InterfaceHelpers.h"
#include "Optimization/Backpropagation.h"
#include "Models/IFeedforwardNetwork.h"

namespace NNS 
{
	namespace Optimization 
	{
		Backpropagation::Backpropagation(ErrorUnit learningRate, ErrorUnit momentumCoeff)
			: learningRate{ learningRate }, momentumCoeff{ momentumCoeff }
		{
			// Nop
		}

		void Backpropagation::Free() const
		{
			delete this;
		}

		void Backpropagation::Initialize(IFeedforwardNetwork& network)
		{
			const auto networkmap = network.GetNetworkLayerMap();
			prevMomentumMatrix.clear();
			prevMomentumMatrix.resize(networkmap.size() - 1);

			for (size_t i = 0; i < networkmap.size() - 1; ++i) /* For each layer ( minus input layer ). */
			{
				prevMomentumMatrix[i].resize(networkmap[i + 1]);

				for (size_t j = 0; j < networkmap[i + 1]; ++j) /* For each neuron. */
				{
					prevMomentumMatrix[i][j].resize(networkmap[i] + 1, 0.0); /* Copy connections number. */
					/* +1 becaue of additional bias */
				}
			}
		}

		bool Backpropagation::OptimizeWeights(IFeedforwardNetwork& network, TrainingErrorState& errorState)
		{
			ErrorUnit correction = 0.0; /* Temporal variable */
			const auto networkmap = network.GetNetworkLayerMap(); /* Obtain network architecture */

			/* For each layer ( minus input layer ). */
			for (size_t i = 0; i < networkmap.size() - 1; ++i)
			{
				/* For each neuron. */
				for (size_t j = 0; j < networkmap[i + 1]; ++j)
				{
					/* For each connection with previous layer + bias */
					for (size_t k = 0; k <= networkmap[i]; ++k)
					{
						/* Calculate weight correction. */
						correction = learningRate * errorState.GetErrorGradient()[i][j][k] + momentumCoeff * prevMomentumMatrix[i][j][k];

						/* Apply the correction for synaptic weight and bias*/
						network.Weight(i + 1, j, k) += correction;

						/* Save correction next iteration. */
						prevMomentumMatrix[i][j][k] = correction;
					}
				}
			}
			return false;
		}
	}
}