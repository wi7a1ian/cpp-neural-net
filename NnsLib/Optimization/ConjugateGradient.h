#pragma once

#include <random>

#include "Types/Collections.h"
#include "Optimization/IWeightOptimizer.h"

namespace NNS 
{
	namespace Optimization
	{
		using namespace NNS::Types;
		using DirectionMatrix = ErrorGradientMatrix;

		/** Training by conjugate gradients.
		* Supervised training for multilayer perceptron networks using conjugate gradients algorithm.
		*/
		class ConjugateGradient : public IWeightOptimizer { 
		public:

			/** Constructor.
			* Parameter list contains three means for escape from conjugate gradient algorithm.
			* @param _errorDeltatolerance convergence indicator. Iteration terminates once a line minimization fails to reduce the error by approximately
			*         this fraction of the actual error.
			* @param maxInternalIter sets limit on the number of iterations  allowed inside conjugate gradient loop.
			* @parem maxRandomRetry sets limit on the number of random directions generated if the directional minimization is not effective.
			*/
			ConjugateGradient(ErrorUnit errorDeltaTolerance = 0.0001, size_t maxInternalIter = 1000, int maxRandomRetry = 5);

			void Free() const override;

			void Initialize(IFeedforwardNetwork& network) override;

			/** Conjugate gradient algorithm.
			* Conjugate gradient algorithm which intelligently choose the search directions for line minimization method.
			* Based on Polak-Ribiere (1971) work, which proves that if our n-dimensional function to minimize ( epoch error ) can be expressed
			* as a quadratic form, then minimizing along the first n search directions 'h' will lead to exact minimum.
			* Neural network error functions are approximately quadratic near local minima thus this method
			* can be expected to converge quickly once it is near the minimum.
			* @see ConjugateGradient::LineMinimization()
			*/
			bool OptimizeWeights(IFeedforwardNetwork& network, TrainingErrorState& errorState) override;

		private:
			
			/** Calculate gamma constant.
			* Used along with work matrix 'g' and search direction 'h' to calculate new search direction.
			* @param tempMatrixG work matrix 'g'.
			* @return calculated gamma.
			*/
			ErrorUnit ComputeGamma(TrainingErrorState& errorState, ErrorGradientMatrix& tempMatrixG);

			/** Calculate new search direction using gamma constant along with work matrix 'g' and search direction 'h'.
			* It computes by adding gamma times the old search direction to the current negative gradient.
			* This method also copies current weight gradient into work matrix 'g' and newly calculated search direction into 'h' and work gradient.
			* @param gamma gamma constant.
			* @param tempMatrixG work matrix 'g'.
			* @param searchDirectionH search direction matrix 'h'.
			*/
			void ComputeNewSearchDirection(TrainingErrorState& errorState, ErrorUnit gamma, ErrorGradientMatrix& tempMatrixG, DirectionMatrix& searchDirectionH);

			/** Line minimization method.
			* Finding the minimum of the error function when the weight variables are constrained to lie along a line ( directional minimization ).
			* Two steps are performed:
			*  - determine the minimum by finding three points such that the middle point is less ( smaller error value ) than the others,
			*  - refine the interval containing the minimum untill we are satisfied with the accuracy.
			* @param startError Error (function value) at starting coefficients.
			* @param maxIterations Upper limit on number of iterations allowed.
			* @param epsilon Small, but greater than machine precision.
			* @param tolerance Brent's tolerance (>= sqrt machine precision).
			*/
			ErrorUnit LineMinimization(IFeedforwardNetwork& network, TrainingErrorState& errorState, ErrorUnit startError, size_t maxIterations, ErrorUnit epsilon, ErrorUnit tolerance);

			/** Method to step out from base.\ Computes new weights appropriately.
			* @param step size.
			* @param direction search direction matrix ( weight gradient ).
			* @param baseWeights base weight matrix.
			*/
			void StepOut(IFeedforwardNetwork& network, ErrorUnit step, DirectionMatrix& direction, WeightMatrix& baseWeights);

			/** Method to make direction gradient be the actual distance moved.
			* This method multiplies the search direction matrix by the specified value.
			* @param step multiplier.
			* @param direction search direction matrix ( weight gradient ).
			*/
			void UpdateDirection(ErrorUnit step, DirectionMatrix& direction);

			/** Reverse the search direction.
			* @param direction search direction matrix ( weight gradient ).
			*/
			void ReverseDirection(DirectionMatrix& direction);

			NetworkLayerMap networkmap;

			size_t maxIterations;
			ErrorUnit errorDeltaTolerance; /**< Iteration terminates once a line minimization fails to reduce the error by approximately this fraction of the actual error. */
			size_t maxInternalIterations; /**< Limit on the number of iterations  allowed inside conjugate gradient loop. */
			int maxRandomRetry; /**< Limit on the number of random directions generated if the directional minimization is not effective. */

			ErrorGradientMatrix tempMatrixG; /**< Work matrix for Polak-Ribiere (1971) ( conjugate gradient ) algorithm. */
			DirectionMatrix searchDirectionH; /**< Generated search directions which are mutually conjugate. */

			std::mt19937 rngEngine; /**< This engine produces randomness out of thin air. */
			std::uniform_real_distribution<ErrorUnit> rngUni01; /**< Uniform distribution in range <0;1> for random number generator. */
		};
	}
}