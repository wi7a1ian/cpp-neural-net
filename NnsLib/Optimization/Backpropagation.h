#pragma once

#include <vector>

#include "Optimization/IWeightOptimizer.h"
#include "Types/Units.h"

namespace NNS 
{
	namespace Optimization 
	{

		using namespace NNS::Types;
		using MomentumMatrix = std::vector<std::vector<std::vector<ErrorUnit>>>;

		/** Backpropagation Training Algorithm.
		* Supervised training for multilayer perceptron networks using backpropagation algorithm ( gradient descent algorithm ).
		* Algorithm contain significant modification to the basic backpropagation method with the addition of a momentum term.
		*/
		class Backpropagation : public IWeightOptimizer {
		public:

			/** Constructor.
			* Parameter list contains two means for escape from gradient descent training.
			* @param learningRate denotes learning coefficient and is static for the whole training process.
			* @param momentumCoeff responsible for additional momentum. We add to currently calculated direction matrix a moderate fraction of the previous one.
			*/
			Backpropagation(ErrorUnit learningRate = 0.25, ErrorUnit momentumCoeff = 0.9);

			void Free() const override;

			/** Pre-training procedure. Used for initializing matrices, etc.
			* Here we initialize momentum matrix. The most significant modification to the basic backpropagation method is the addition of a momentum term.
			* Rather than letting our search direction wildly trash about as the gradient changes, we impose a momentum on it.
			* Each new search direction is computed as a weighted sum of the current gradient and the previous search direction.
			*/
			void Initialize(IFeedforwardNetwork& network) override;

			bool OptimizeWeights(IFeedforwardNetwork& network, TrainingErrorState& errorState) override;

		protected:
			MomentumMatrix prevMomentumMatrix; /**< Previous momentum values */

			ErrorUnit learningRate; /**< gradient multiplier */
			ErrorUnit momentumCoeff; /**< momentum coefficient */

		private:
			// None
		};
	}
}