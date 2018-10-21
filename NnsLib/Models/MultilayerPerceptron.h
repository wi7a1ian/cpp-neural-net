#pragma once

#include "Models/FeedforwardNetworkBase.h"
#include "Common/ActivationFunctions.h"
#include "Types/Collections.h"


namespace NNS 
{
	namespace Models 
	{

		using namespace NNS::Types;
		using namespace NNS::Activation;

		class MultilayerPerceptron : public FeedforwardNetworkBase, public IBiased
		{
		public:
			explicit MultilayerPerceptron(std::initializer_list<int> networkLayerMap);

			bool ComputeOutput(InputLayer const& inputLayer) override;

			SignalUnit GetActivationDerivative(int layerId, int neuronId) const override;

			WeightUnit& Bias(int layerId, int neuronId) override;
			void SetBiasForAll(WeightUnit value = 1.0) override;

			virtual void Rebuild() override;

		private:
			LogisticActivationFunction<SignalUnit> activationFunction; // TODO: make configurable
		};
	}
}