#include "pch.h"
#include "Training/TrainingErrorState.h"

namespace NNS 
{
	namespace Training 
	{

		TrainingErrorState::TrainingErrorState(IFeedforwardNetwork& network, const TrainingDataSet& trainingData)
			: network{ network }, trainingData{ trainingData }, networkmap{ network.GetNetworkLayerMap() }
		{
			SetErrorComputationMethod(ErrorCalculationMethod::MeanSquareError);
			InitializeMatrices();
		};

		void TrainingErrorState::SetErrorComputationMethod(ErrorCalculationMethod method)
		{
			errorMethod = method;
		}

		ErrorVector const& TrainingErrorState::GetErrorVector()
		{
			return epochErrorVector;
		}

		void TrainingErrorState::UpdateErrorVector(ErrorUnit error)
		{
			epochErrorVector.push_back(error);
		}

		void TrainingErrorState::InitializeMatrices()
		{
			errorGradient.clear();
			errorDelta.clear();

			errorGradient.resize(networkmap.size() - 1);
			errorDelta.resize(networkmap.size() - 1);

			for (size_t i = 0; i < networkmap.size() - 1; ++i) /* For each layer ( minus input layer ). */
			{
				errorGradient[i].resize(networkmap[i + 1]);
				errorDelta[i].resize(networkmap[i + 1], 0.0);

				for (size_t j = 0; j < networkmap[i + 1]; ++j) /* For each neuron. */
				{
					errorGradient[i][j].resize(networkmap[i] + 1, 0.0); /* Copy connections number. */
																		/* +1 becaue of additional bias */
				}
			}
		}

		void TrainingErrorState::ZeroErrorGradient()
		{
			for (size_t i = 0; i < networkmap.size() - 1; ++i) /* For each layer ( minus input layer ). */
				for (size_t j = 0; j < networkmap[i + 1]; ++j) /* For each neuron. */
					for (size_t k = 0; k < networkmap[i] + 1; ++k) /* For each connection + bias. */
						errorGradient[i][j][k] = 0.0;
		}

		ErrorUnit TrainingErrorState::ComputeEpochError(bool computeGradient)
		{
			ErrorUnit error{};

			if (computeGradient)
			{
				ZeroErrorGradient();
			}

			// For each presentation in epoch.
			for (const auto& trainingDataStep : trainingData)
			{
				error += ComputeError(trainingDataStep.first, trainingDataStep.second);

				if (computeGradient)
				{
					ComputeErrorGradient(trainingDataStep.first, trainingDataStep.second);
				}
			}

			assert((static_cast<ErrorUnit>(trainingData.size())) != 0);

			return error / (static_cast<ErrorUnit>(trainingData.size()));
		}

		ErrorUnit TrainingErrorState::ComputeEpochGradient()
		{
			return ComputeEpochError(true);
		}

		ErrorGradientMatrix& TrainingErrorState::GetErrorGradient()
		{
			return errorGradient;
		}

		ErrorUnit TrainingErrorState::ComputeError(InputLayer const& iutputLayer, OutputLayer const& desiredOutputLayer)
		{
			switch (errorMethod)
			{
			case ErrorCalculationMethod::LogMeanSquareError:
			{
				return this->ComputeLogMeanSquareError(iutputLayer, desiredOutputLayer);
			}
			case ErrorCalculationMethod::MeanSquareError:
			default:
			{
				return this->ComputeMeanSquareError(iutputLayer, desiredOutputLayer);
			}
			}
		}

		ErrorUnit TrainingErrorState::ComputeMeanSquareError(InputLayer const& iutputLayer, OutputLayer const& desiredOutputLayer)
		{
			ErrorUnit error{};

			if (!network.ComputeOutput(iutputLayer))
			{
				return error;
			}

			for (size_t i = 0; i < desiredOutputLayer.size(); ++i)
			{
				error += pow(desiredOutputLayer[i] - network.GetOutputActivation(i), 2.0);
			}

			error = (error / static_cast<ErrorUnit>(desiredOutputLayer.size()));
			return error;
		}

		ErrorUnit TrainingErrorState::ComputeLogMeanSquareError(InputLayer const& iutputLayer, OutputLayer const& desiredOutputLayer)
		{
			return log(ComputeMeanSquareError(iutputLayer, desiredOutputLayer));
		}

		void TrainingErrorState::ComputeErrorGradient(InputLayer const& iutputLayer, OutputLayer const& desiredOutputLayer)
		{
			SignalUnit delta, sum;

			for (size_t i = networkmap.size() - 1; i > 0; --i) /* For each layer ( minus input layer ). */
			{
				for (size_t j = 0; j < networkmap[i /* current layer */]; ++j) /* For each neuron. */
				{
					if (i == networkmap.size() - 1) /* Calculating delta for the output layer */
					{
						delta = (desiredOutputLayer[j] - network.GetOutputActivation(j)) * network.GetActivationDerivative(i, j);
					}
					else
					{ /* Calculating delta for hidden layers */
						sum = 0.0;

						for (size_t k = 0; k < networkmap[i + 1/* next layer */]; ++k) /* For each neuron from next layer. */
							sum += errorDelta[i /* next layer */][k] * network.Weight(i + 1 /* next layer */, k /* k'th neuron */, j /* connection to this layer */);

						delta = sum * network.GetActivationDerivative(i, j);
					}
					errorDelta[i - 1][j] = delta; /* Save error delta. */
					
					/* Calculating partial derivative of the error. */
					for (size_t k = 0; k < networkmap[i - 1]; ++k)
						errorGradient[i - 1 /* current layer */][j][k] += delta * network.GetActivation(i - 1 /* previus layer */, k);

					errorGradient[i - 1 /* current layer */][j][networkmap[i - 1]] += delta; /* Bias activation is always equal to 1.*/
				}
			}
		}
	}
}