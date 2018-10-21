#pragma once
#include <vector>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::Matrix;
using Eigen::Dynamic;

#include "Types/Units.h"

namespace NNS 
{
	namespace Types 
	{
		using std::vector;
		using std::size_t;
		using std::pair;

		using ActivationVector		= Matrix<SignalUnit, Dynamic, 1>;
		using InputLayer			= ActivationVector;
		using OutputLayer			= ActivationVector;
		using ActivationMatrix		= vector<ActivationVector>; // TODO: Eigen::SparseMatrix<>
		using NetworkLayerMap		= vector<size_t>;

		using WeightVector			= Matrix<SignalUnit, Dynamic, 1>;
		using WeightMatrix			= vector<vector<WeightVector>>;  // TODO: Eigen::SparseMatrix<>

		using ErrorVector			= vector<ErrorUnit>;
		using TrainingDataSet		= vector<pair<InputLayer, OutputLayer>>;
		using ErrorGradientMatrix	= vector<vector<vector<ErrorUnit>>>;  // TODO: Eigen::SparseMatrix<>
		using ErrorDeltaMatrix		= vector<vector<ErrorUnit>>;
	} 
}