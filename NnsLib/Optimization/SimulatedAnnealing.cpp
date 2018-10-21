#include "pch.h"
#include "Optimization/SimulatedAnnealing.h"

namespace NNS 
{
	namespace Optimization 
	{
		SimulatedAnnealing::SimulatedAnnealing(SimulatedAnnealingConfig cfg)
			: Config{ cfg }
		{
			rngEngine.seed(static_cast<int>(time(0)));
		}

		void SimulatedAnnealing::Free() const
		{
			delete this;
		}

		void SimulatedAnnealing::Initialize(IFeedforwardNetwork& network)
		{
			// Nop
		}

		bool SimulatedAnnealing::OptimizeWeights(IFeedforwardNetwork& network, TrainingErrorState& errorState)
		{
			ComputeSimulatedAnnealing(network, errorState);
			return true;
		}

		void SimulatedAnnealing::ComputeSimulatedAnnealing(IFeedforwardNetwork& network, TrainingErrorState& errorState)
		{
			bool improved; /* True if we improved. */
			size_t seed, best_seed;
			ErrorUnit error, best_error; /* Current error and best achieved error. */
			WeightMatrix best_weights; /* Work area used to keep best network */
			const auto networkmap = network.GetNetworkLayerMap();

			/* Configure random number generation for simulated annealing. */
			if (Config.perturbationDistribution == RandomDistributionMethod::Normal)
			{
				rngGaussianParams = new std::normal_distribution<ErrorUnit>::param_type(0.0, Config.perturbationVariance);
				rngGaussian.param((*rngGaussianParams));
			}

			best_weights = network.GetWeightMatrix(); /* Current weights are best so far. */
			best_error = errorState.ComputeEpochError();

			auto temperature = Config.startTemperature;
			auto temperature_mult = exp(log(Config.stopTemperature / Config.startTemperature) / (Config.temperatureNumber - 1));

			for (size_t i = 0; i < static_cast<size_t>(Config.temperatureNumber); ++i) /* Iterate over number of temperatures. */
			{
				improved = false;

				for (size_t j = 0; j < Config.temperatureIters; ++j) /* Iterate over number of iterations per temperature. */
				{
					/* Instead of copying weight matrix in case of success, we will just save best seed for random number generator */
					/* and recreate that weight matrix later. Thanks to this we reduce computation time and memory usage. */
					seed = rngUniInt(rngEngine); /* Get a random seed. */
					rngEngine.seed(seed);

					if (Config.perturbationDistribution == RandomDistributionMethod::Normal)
						rngGaussian.reset();

					ComputeWeightsPerturbation(network, best_weights, temperature); /* Randomly perturb about best. */

					error = errorState.ComputeEpochError();

					if (error < best_error) /* If this iteration improved then update the best record. */
					{
						best_error = error;
						best_seed = seed; /* Save seed to recreate it. */
						//best_weights = _network->GetWeightMatrix();
						improved = true;

						if (best_error <= Config.errorThreshold) /* Stop if we reached the error threshold. */
							break;

						j -= Config.setback; /* It often pays to keep going at this temperature if we are still improving. */
						if (j < 0)
							j = 0;
					}
				}

				if (improved) /* If this temperature saw improvement. */
				{
					rngEngine.seed(best_seed); /* Reassign best seed. */
					rngGaussian.reset();
					ComputeWeightsPerturbation(network, best_weights, temperature); /* Recreate best weights. */
					best_weights = network.GetWeightMatrix(); /* New best weights. */
				}

				if (best_error <= Config.errorThreshold) /* Stop if we reached the error threshold. */
					break;

				/* We may break computation here if we need to. */
				/*if (trainer.IsTrainingAborted())
					break;*/

				temperature *= temperature_mult; /* Reduce temp for next pass. */
			}

			/* Apply the best weights we got into the multilayer perceptron. */
			for (size_t i = 1; i < static_cast<int>(networkmap.size()); ++i) /* For each layer ( minus input layer ). */
				for (size_t j = 0; j < networkmap[i]; ++j) /* For each neuron. */
					for (size_t k = 0; k < networkmap[i - 1] + 1; ++k) /* For each connection + bias. */
						network.Weight(i, j, k) = best_weights[i - 1][j][k];
		}

		void SimulatedAnnealing::ComputeWeightsPerturbation(IFeedforwardNetwork& network, WeightMatrix& center, ErrorUnit temperature) throw()
		{
			const auto networkmap = network.GetNetworkLayerMap();

			/* We reduced the periodicallity of random numbers by using mt19937 pseudo-random number generator. */
			/* It is derivative of mersenne twister engine and is better than linear congruential engine. */
			/* We also may use normal distribution ( gaussian ) instead of uniform distribution. */

			for (size_t i = 1; i < networkmap.size(); ++i) /* For each layer ( minus input layer ). */
			{
				for (size_t j = 0; j < networkmap[i]; ++j) /* For each neuron. */
				{
					for (size_t k = 0; k < networkmap[i - 1] + 1; ++k) /* For each connection + bias. */
					{
						if (Config.perturbationDistribution == RandomDistributionMethod::Normal)
						{
							network.Weight(i, j, k) = center[i - 1][j][k] + temperature * rngGaussian(rngEngine);
						}
						else if (Config.perturbationDistribution == RandomDistributionMethod::Uniform)
						{
							network.Weight(i, j, k) = center[i - 1][j][k] + temperature * (1 - 2 * rngUni01(rngEngine));
						}
					}
				}
			}
		}
	}
}