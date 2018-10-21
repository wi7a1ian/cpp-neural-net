#pragma once

#include <memory>

#include "Common/IBase.h"
#include "Models/IFeedforwardNetwork.h"
#include "Training/TrainingErrorState.h"

namespace NNS 
{
	namespace Optimization 
	{
		using NNS::Models::IFeedforwardNetwork;
		using NNS::Training::TrainingErrorState;

		class IWeightOptimizer : public IBase
		{
		public:
			using Ptr = std::unique_ptr<IWeightOptimizer, SDeleter>;
		public:
			virtual void Initialize(IFeedforwardNetwork& network) = 0;
			virtual bool OptimizeWeights(IFeedforwardNetwork& network, TrainingErrorState& errorState) = 0;
		};
	}
}