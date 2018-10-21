#include "pch.h"
#include "Models/FeedforwardNetworkBase.h"

namespace NNS 
{
	namespace Models 
	{

		FeedforwardNetworkBase::FeedforwardNetworkBase(std::initializer_list<int> networkLayerMap)
		{
			if (networkLayerMap.size() < 2)
			{ 
				throw std::invalid_argument("Invalid network size");
			}
			else
			{
				for (const auto& layerSize : networkLayerMap)
				{
					activationMatrix.push_back(ActivationVector::Zero(layerSize));
				}
			}
		}

		void FeedforwardNetworkBase::Free() const
		{
			delete this;
		}

		void FeedforwardNetworkBase::Rebuild()
		{
			weightMatrix = WeightMatrix{ activationMatrix.size() - 1 };

			// For each layer ( minus input layer ).
			for (size_t i = 1; i < activationMatrix.size(); ++i)
			{
				weightMatrix[i - 1].resize(activationMatrix[i].size());
				// For each neuron.
				for (size_t j = 0; j < activationMatrix[i].size(); ++j)
				{
					// Set previous layer size
					weightMatrix[i - 1][j] = WeightVector::Zero(activationMatrix[i - 1].size());
				}
			}
				
			isWeightMagLimited = false;
		}

		WeightUnit& FeedforwardNetworkBase::Weight(int layerId, int neuronId, int connectionId)
		{
			isWeightMagLimited = true; /* In case we need to limit weight magnitude. */
			return weightMatrix[layerId - 1][neuronId][connectionId];
		}

		SignalUnit const& FeedforwardNetworkBase::GetActivation(int layerId, int neuronId) const
		{
			return activationMatrix[layerId][neuronId];
		}

		SignalUnit const& FeedforwardNetworkBase::GetOutputActivation(int neuronId) const
		{
			return activationMatrix.back()[neuronId];
		}

		NetworkLayerMap FeedforwardNetworkBase::GetNetworkLayerMap() const throw()
		{
			NetworkLayerMap layers(activationMatrix.size());

			for (size_t i = 0; i < activationMatrix.size(); ++i)
				layers[i] = activationMatrix[i].size();

			return layers;
		}

		void FeedforwardNetworkBase::SetWeightMagnitudeLimit(WeightUnit limit)
		{
			weightMagnitudeLimit = fabs(limit);
			if (weightMagnitudeLimit > 0.0)
			{
				for (size_t i = 0; i < weightMatrix.size(); ++i) /* Each layer, except first */
				{
					for (size_t j = 0; j < weightMatrix[i].size(); ++j) /* Each neuron */
					{
						for (size_t k = 0; k < weightMatrix[i][j].size(); ++k) /* Each previous connection */
						{
							if (fabs(weightMatrix[i][j][k]) > weightMagnitudeLimit)
							{
								if (weightMatrix[i][j][k] > 0.0)
									weightMatrix[i][j][k] = weightMagnitudeLimit;
								else
									weightMatrix[i][j][k] = -weightMagnitudeLimit;
							}
						}
					}
				}
			}

			isWeightMagLimited = false;
		}

		ActivationMatrix const& FeedforwardNetworkBase::GetActivationMatrix() const
		{
			return activationMatrix;
		}

		WeightMatrix& FeedforwardNetworkBase::GetWeightMatrix()
		{
			if (isWeightMagLimited && weightMagnitudeLimit != 0.0)
			{
				/* check weights magnitude for correctness */
				SetWeightMagnitudeLimit(weightMagnitudeLimit);
			}

			return weightMatrix;
		}
	} 
}