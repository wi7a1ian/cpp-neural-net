#pragma once

#include "Models/FeedforwardNetworkBase.h"
#include "Types/Collections.h"

namespace NNS 
{
	namespace Models 
	{
		using namespace NNS::Types;

		class KohonenNetwork : public FeedforwardNetworkBase {
		public:

			KohonenNetwork() = delete;
			KohonenNetwork(int inputLayerSize, int outputLayerSize);

			KohonenNetwork(const KohonenNetwork&) = delete;
			KohonenNetwork& operator=(const KohonenNetwork&) = delete;

			bool ComputeOutput(InputLayer const& inputLayer) throw() override;
			
			SignalUnit GetActivationDerivative(int layerId, int neuronId) const override;

		protected:
			void InitializeKohonen();
		};
	}
}