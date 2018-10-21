#include "pch.h"
#include "Models/MultilayerPerceptron.h"


namespace NNS 
{
	namespace Models 
	{

		MultilayerPerceptron::MultilayerPerceptron(std::initializer_list<int> networkLayerMap)
			: FeedforwardNetworkBase(networkLayerMap) 
		{ 
			Rebuild();
		};


		void MultilayerPerceptron::Rebuild()
		{
			weightMatrix = WeightMatrix{ activationMatrix.size() - 1 };

			for (size_t i = 1; i < activationMatrix.size(); ++i) /* For each layer ( minus input layer ). */
			{
				weightMatrix[i - 1].resize(activationMatrix[i].size());
				for (size_t j = 0; j < activationMatrix[i].size(); ++j) /* For each neuron. */
				{
					weightMatrix[i - 1][j] = WeightVector::Zero(activationMatrix[i - 1].size() + 1); /* Previous layer size + bias */
					weightMatrix[i - 1][j].tail(1)[0] = 1.0; /* Bias is always equal to 1.0 */
				}
			}
				
			isWeightMagLimited = false;
		}

		bool MultilayerPerceptron::ComputeOutput(InputLayer const& inputLayer)
		{
			if (weightMatrix.empty() || activationMatrix.front().size() != inputLayer.size())
				return false;
			
			activationMatrix.front() = inputLayer;

			if (isWeightMagLimited && weightMagnitudeLimit != 0.0)
				SetWeightMagnitudeLimit(weightMagnitudeLimit); /* check weights magnitude for correctness */

			for (size_t i = 1; i < activationMatrix.size(); ++i)  /* Each layer, except first */
			{
				auto& prevLayer = activationMatrix[i - 1];
				for (size_t j = 0; j < activationMatrix[i].size(); ++j) /* Each neuron */
				{
					auto& weightVect = weightMatrix[i - 1][j];
					auto& weightVectWithoutBias = weightVect.head(prevLayer.size());
					auto& bias = weightVect[weightVect.size() - 1];

					activationMatrix[i][j] = activationFunction(bias + prevLayer.dot(weightVectWithoutBias));
				}
			}

			return true;
		}

		SignalUnit MultilayerPerceptron::GetActivationDerivative(int layerId, int neuronId) const
		{
			return activationFunction.Deriv(GetActivation(layerId, neuronId));
		}

		WeightUnit& MultilayerPerceptron::Bias(int layerId, int neuronId)
		{
			isWeightMagLimited = true;
			return weightMatrix[layerId - 1][neuronId].tail(1)[0];
		}

		void MultilayerPerceptron::SetBiasForAll(WeightUnit value)
		{
			for (size_t i = 0; i < weightMatrix.size(); ++i)
			{	/* Each layer, except first */
				for (size_t j = 0; j < weightMatrix[i].size(); ++j)
				{	/* Each neuron */
					weightMatrix[i][j].tail(1)[0] = value;
				}
			}
		}
	}
}