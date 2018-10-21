#pragma once

#include <ctime>
#include <random>

#include "Initialization/IWeightInitializer.h"

namespace NNS {
	namespace Initialization {
		using namespace NNS::Types;
		using NNS::Models::IFeedforwardNetwork;

		class RandomWeightInitializer final : public IWeightInitializer {
		public:

			explicit RandomWeightInitializer(WeightUnit magnitude = 0.5);
			void Free() const override;

			void InitializeWeights(IFeedforwardNetwork& network) override;

		private:
			WeightUnit randomMagnitude; /**< Maximal magnitude value in case of random initial weight generation. */
		};

	}
}