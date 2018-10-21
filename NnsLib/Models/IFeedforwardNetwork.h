#pragma once

#include <memory>

#include "Common/IBase.h"
#include "Types/Collections.h"

namespace NNS
{
	namespace Models
	{
		using namespace NNS::Types;

		class IFeedforwardNetwork : public IBase
		{
		public:
			using Ptr = std::unique_ptr<IFeedforwardNetwork, SDeleter>;
		public:
			virtual NetworkLayerMap GetNetworkLayerMap() const = 0;
			virtual bool ComputeOutput(InputLayer const& inputLayer) = 0;
			virtual void Rebuild() = 0;

			virtual ActivationMatrix const& GetActivationMatrix() const = 0;
			virtual SignalUnit const& GetActivation(int layerId, int neuronId) const = 0;
			virtual SignalUnit GetActivationDerivative(int layerId, int neuronId) const = 0;
			virtual SignalUnit const& GetOutputActivation(int neuronId) const = 0;

			virtual WeightMatrix& GetWeightMatrix() = 0;
			virtual WeightUnit& Weight(int layerId, int neuronId, int connectionId) = 0;
		};

		// Additional behaviours

		class IBiased
		{
		public:
			virtual WeightUnit& Bias(int layerId, int neuronId) = 0;
			virtual void SetBiasForAll(WeightUnit value = 1.0) = 0;
		};
	}
}