#pragma once

#include "Models/IFeedforwardNetwork.h"

namespace NNS 
{
	namespace Models 
	{
		using namespace NNS::Types;

		class FeedforwardNetworkBase : public IFeedforwardNetwork
		{
		public:
			explicit FeedforwardNetworkBase(std::initializer_list<int> networkLayerMap);

			void Free() const override;

			NetworkLayerMap GetNetworkLayerMap() const override;
			void Rebuild() override;

			ActivationMatrix const& GetActivationMatrix() const override;
			SignalUnit const& GetActivation(int layerId, int neuronId) const override;
			SignalUnit const& GetOutputActivation(int neuronId) const override;

			WeightMatrix& GetWeightMatrix() override;
			WeightUnit& Weight(int layerId, int neuronId, int connectionId) override;

			void SetWeightMagnitudeLimit(WeightUnit limit = 5.0);

		protected:
			WeightMatrix weightMatrix;
			ActivationMatrix activationMatrix;

			WeightUnit weightMagnitudeLimit{ 0.0 };
			bool isWeightMagLimited{ false };
		};
	}
}