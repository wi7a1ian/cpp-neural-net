#pragma once

#include <memory>

#include "Types/Units.h"
#include "Types/Collections.h"
#include "Models/IFeedforwardNetwork.h"

namespace NNS 
{
	namespace Training 
	{

		using std::vector;
		using std::pair;
		using std::size_t;

		using namespace NNS::Types;
		using namespace NNS::Models;

		enum class ErrorCalculationMethod : unsigned int
		{
			MeanSquareError = 0,
			LogMeanSquareError
		};

		class TrainingErrorState
		{
		public:
			using Ptr = std::unique_ptr<TrainingErrorState>;

			TrainingErrorState(IFeedforwardNetwork& network, const TrainingDataSet& trainingData);

			void SetErrorComputationMethod(ErrorCalculationMethod method);
			void InitializeMatrices();
			void ZeroErrorGradient();

			ErrorVector const& GetErrorVector();
			void UpdateErrorVector(ErrorUnit error);

			ErrorUnit ComputeEpochError(bool computeGradient = false);
			ErrorUnit ComputeEpochGradient();

			ErrorGradientMatrix& GetErrorGradient();

		protected:
			ErrorUnit ComputeError(InputLayer const& iutputLayer, OutputLayer const& desiredOutputLayer);
			ErrorUnit ComputeMeanSquareError(InputLayer const& iutputLayer, OutputLayer const& desiredOutputLayer);
			ErrorUnit ComputeLogMeanSquareError(InputLayer const& iutputLayer, OutputLayer const& desiredOutputLayer);

			// Calculate partial error value as well as objective function gradient.
			void ComputeErrorGradient(InputLayer const& iutputLayer, OutputLayer const& desiredOutputLayer);

		private:
			ErrorGradientMatrix errorGradient;
			ErrorDeltaMatrix errorDelta; // Matrix with Partial derivative of the error.

			IFeedforwardNetwork& network;
			const TrainingDataSet& trainingData;
			const NetworkLayerMap networkmap;

			ErrorCalculationMethod errorMethod;
			ErrorVector epochErrorVector; /**< Error obtained after computing each presentation. */
			ErrorVector::iterator epochErrorVectorIter; /**< Iterator for _epochErrorVector. */
		};
	} 
}