#pragma once

#include <memory>

#include "Common/IBase.h"
#include "Models/IFeedforwardNetwork.h"
#include "Optimization/IWeightOptimizer.h"

namespace NNS
{
	namespace Training
	{
		using NNS::Models::IFeedforwardNetwork;
		using NNS::Optimization::IWeightOptimizer;

		class ITrainingAlgorithm : public IBase
		{
		public:
			using Ptr = std::unique_ptr<ITrainingAlgorithm, SDeleter>;
		public:
			virtual void Train(IFeedforwardNetwork& network, TrainingDataSet const& trainingData) = 0;
			virtual void SetEludingLocalMinimaMethod(IWeightOptimizer* optimizer) = 0; // TODO: more than one?
			virtual void AbortTraining() = 0;
			virtual bool IsTrainingAborted() const = 0;
		};
	}
}