#pragma once

#include <ctime>
#include <random>

#include "Optimization/IWeightOptimizer.h"

namespace NNS 
{
	namespace Optimization 
	{

		using namespace NNS::Types;
		using NNS::Models::IFeedforwardNetwork;
		using NNS::Types::WeightMatrix;
		using NNS::Training::TrainingErrorState;

		enum class RandomDistributionMethod : unsigned int
		{
			Uniform = 0,
			Normal
		};

		struct SimulatedAnnealingConfig final 
		{
			// Standard deviation of the rendom perturbation used first. 
			// Should be set to several times the maximum expected distance between the starting guess and the global minimum point.
			ErrorUnit startTemperature{ 1.0f }; 
			// This is the final standard deviation. 
			// It should be of the order of magnitude of the desired accuracy in the location of the best point. 
			ErrorUnit stopTemperature{ 0.01f };
			
			// Stop annealing if error drops this low.
			ErrorUnit errorThreshold{ 0.00001f };
			// Number of temperatures ( 2-3 for init, 4-5 for eluding local minima). 
			// Larger values should be used if there is a great difference in start and stop temperatures.
			int temperatureNumber{ 5 };
			// Iterations per temperature. Should be set as large as possible. 
			// Setback param may cause exceed of this parameter.
			size_t temperatureIters{ 100 };			
			// Set back iteration counter if improvement. 
			// Typical value for this param is about half of temperatureIters.
			size_t setback{ 30 };

			// Distribution of random numbers generated in perturbation of temperatures.s
			RandomDistributionMethod perturbationDistribution{ RandomDistributionMethod::Normal };
			// Variance size of random numbers generated in perturbation of temperatures.
			ErrorUnit perturbationVariance{ 0.5 };
		};

		class SimulatedAnnealing final : public IWeightOptimizer {
		public:
			const SimulatedAnnealingConfig Config;

			explicit SimulatedAnnealing(SimulatedAnnealingConfig config);

			void Free() const override;

			void Initialize(IFeedforwardNetwork& network) override;
			bool OptimizeWeights(IFeedforwardNetwork& network, TrainingErrorState& errorState) override;

		private:
			/** Eluding local minima by means of simulated annealing.
			* Simple yet effective method for avoiding local minima, as well as escaping from them if necessary.
			* Simulated annealing can be performed by randomly perturbing the weights and keeping track of the lowest error value.
			* After many tries the weights that produce the best (lowest) error is designated to be the center about which perturbation will take place for the next temperature.
			* The temperature is then reduced and new tries are done.
			* @param config set of configuration parameters for simulated annealing algorithm.
			*/
			void ComputeSimulatedAnnealing(IFeedforwardNetwork& network, TrainingErrorState& errorState);

			/** Subroutine for simulated annealing algorithm randomly perturbing the weights.
			* @param center Center around which we will perform perturbation.
			* @param temperature Temperature magnitude for perturbations.
			*/
			void ComputeWeightsPerturbation(IFeedforwardNetwork& network, WeightMatrix& center, ErrorUnit temperature);

			std::mt19937 rngEngine; /**< This engine produces randomness out of thin air. */
			std::uniform_real<ErrorUnit> rngUni01; /**< Uniform distribution in range <0;1> for random number generator. */
			std::uniform_int_distribution<int> rngUniInt; /**< Uniform distribution in range <0;MAX INT> for random number generator. */
			std::normal_distribution<ErrorUnit> rngGaussian; /**< Normal (gaussian) distribution for random number generator. */
			std::normal_distribution<ErrorUnit>::param_type* rngGaussianParams; /**< Parameters (mean, variance) for normal distribution. */
		};



	}
}