#pragma once

// Our project's .h files.
#include <atomic>

#include "Types/Units.h"
#include "Types/Collections.h"
#include "Common/ActivationFunctions.h"
#include "Models/IFeedforwardNetwork.h"
#include "Optimization/IWeightOptimizer.h"
#include "Optimization/ConjugateGradient.h"
#include "Optimization/Backpropagation.h"
#include "Training/TrainingErrorState.h"
#include "Training/ITrainingAlgorithm.h"

namespace NNS 
{
	namespace Training 
	{
		using std::vector;
		using std::pair;
		using std::size_t;

		using namespace NNS::Types;
		using namespace NNS::Models;
		using namespace NNS::Optimization;
		using NNS::Activation::ActivationFunctionPtr;

		class SupervisedTraining : public ITrainingAlgorithm
		{
		public:
			/** Constructor.
			* Parameter list contains two means for escape from algorithm.
			* @param maxIterations sets limit on the number of iterations allowed.
			* @param errorThreshold signal convergence if the actual error drops this low ( usually set to 0 ).
			*/
			SupervisedTraining(IWeightOptimizer& algorithm, size_t maxIterations = 1000, ErrorUnit errorThreshold = 0.05);

			void Free() const override;

			/** Training procedure.
			* The goal of the training process is to find the set of weight values that will cause the output from the neural network to match the actual target values as closely as possible.
			* There are several issues involved in designing and training a multilayer perceptron network:
			*  - Selecting how many hidden layers to use in the network.
			*  - Deciding how many neurons to use in each hidden layer.
			*  - Finding a globally optimal solution that avoids local minima.
			*  - Converging to an optimal solution in a reasonable period of time.
			*  - Validating the neural network to test for overfitting.
			*/
			void Train(IFeedforwardNetwork& network, TrainingDataSet const& trainingData) override;

			/** Adjust method for eluding local minima
			* @param method target method
			*/
			void SetEludingLocalMinimaMethod(IWeightOptimizer* optimizer) override;

			/** Inform algorithm to break as soon as possible.
			* Calcutation won't be canceled but only not finalized. Algorithm have few check points,
			* where it can decide if breaking is possible, so executing this method won't terminate the algorithm at exac same time.
			*/
			void AbortTraining() override;
			
			/** Check if calculation was aborted.
			* May be still in progress.
			*/
			bool IsTrainingAborted() const override;

		protected:
			// Selected optimizer for elusion of local minimum.
			IWeightOptimizer* elmAlgorithm{ nullptr };
			IWeightOptimizer& trainingAlgorithm;

			size_t maxIterations;
			ErrorUnit errorThreshold;
			std::atomic<bool> isTrainingAborted;

			TrainingErrorState::Ptr errorState;
		};
	}
}