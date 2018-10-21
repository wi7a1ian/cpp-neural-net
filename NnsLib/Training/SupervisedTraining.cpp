#include "pch.h"
#include "Training/SupervisedTraining.h"

namespace NNS 
{
	namespace Training 
	{

		SupervisedTraining::SupervisedTraining(IWeightOptimizer& algorithm, size_t maxIterations, ErrorUnit errorThreshold)
			: maxIterations{ maxIterations }, errorThreshold{ errorThreshold }, trainingAlgorithm{ algorithm }
		{
		}

		void SupervisedTraining::Free() const
		{
			delete this;

		}
		void SupervisedTraining::Train(IFeedforwardNetwork& network, TrainingDataSet const& trainingData)
		{
			assert(trainingData.size() > 0);

			if (trainingData.size() == 0)
			{
				return;
			}

			isTrainingAborted = false;

			errorState = std::make_unique<TrainingErrorState>(network, trainingData);

			trainingAlgorithm.Initialize(network);

			bool is_completed = false;
			for (size_t i = 0; i < maxIterations; ++i)
			{
				const auto error = errorState->ComputeEpochGradient();
				
				// TODO: save best error and weight combination
				errorState->UpdateErrorVector(error);
				

				if (error <= errorThreshold || is_completed || isTrainingAborted)
				{
					/* If error is small enought, then we can break learning procedure. */
					break;
				}

				is_completed = trainingAlgorithm.OptimizeWeights(network, *errorState);

				if (is_completed || isTrainingAborted)
				{
					/* If TrainingProcedure forces us to finish ( either because of failure or just because the algorithm decided to stop */
					continue;
				}

				if (elmAlgorithm != nullptr)
				{
					/* Eluding local minima in learning. */
					elmAlgorithm->OptimizeWeights(network, *errorState);
				}
			}
		}

		void SupervisedTraining::SetEludingLocalMinimaMethod(IWeightOptimizer* optimizer)
		{
			assert(optimizer != nullptr);
			elmAlgorithm = optimizer;
		}

		void SupervisedTraining::AbortTraining()
		{
			isTrainingAborted = true;
		}

		bool SupervisedTraining::IsTrainingAborted() const
		{
			return isTrainingAborted;
		}
	}
}