#include "pch.h"
#include "Models/KohonenNetwork.h"

namespace NNS 
{
	namespace Models 
	{

		KohonenNetwork::KohonenNetwork(int inputLayerSize, int outputLayerSize)
			: FeedforwardNetworkBase({ inputLayerSize, outputLayerSize })
		{
			InitializeKohonen();
		}

		void KohonenNetwork::InitializeKohonen()
		{
			Rebuild();
		}

		bool KohonenNetwork::ComputeOutput(InputLayer const& inputLayer)
		{
			if (weightMatrix.empty() || activationMatrix.front().size() != inputLayer.size())
				return false;

			static const auto networkmap = GetNetworkLayerMap(); /* Obtain network architecture */

			activationMatrix.front() = inputLayer;

			if (isWeightMagLimited && weightMagnitudeLimit != 0.0)
				SetWeightMagnitudeLimit(weightMagnitudeLimit);


			for (size_t i = 1; i < networkmap.size(); ++i)  /* Each layer, except first */
			{
				auto& currLayer = activationMatrix[i];
				auto& prevLayer = activationMatrix[i - 1];

				for (size_t j = 0; j < networkmap[i]; ++j) /* Each neuron */
				{
					SignalUnit activationSum = 0;
					for (size_t k = 0; k < networkmap[i - 1]; ++k) /* Each previous connection */
					{
						activationSum += prevLayer[k] * weightMatrix[i - 1][j][k];
					}
					currLayer[j] = activationSum;
				}
			}

			return true;
		}

		SignalUnit KohonenNetwork::GetActivationDerivative(int layerId, int neuronId) const 
		{
			return GetActivation(layerId, neuronId);
		}
	}
}