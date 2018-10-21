#include "pch.h"

namespace NNSLibTest
{
	using namespace NNS::Models;
	using namespace NNS::Training;
	using namespace NNS::Optimization;
	using namespace NNS::Activation;
	using namespace NNS::Initialization;

	using std::vector;
	using std::shared_ptr;
	using std::unique_ptr;
	using std::make_unique;

	TEST(BackpropagationTests, Xor2to1Problem)
	{
		// given
		MultilayerPerceptron network{ 2, 3, 1 };
		RandomWeightInitializer weight_init{ 0.5 };
		IWeightOptimizer::Ptr algorithm(new Backpropagation(0.25, 0.9));
		SupervisedTraining trainer(*algorithm, 10000, 0.001);
		auto training_set = testHelpers::ReadTrainingDataSet("C:\\Repos\\NNSimulator\\NnsLib.Tests\\TestData\\xor_i2_o1.txt");

		// when
		weight_init.InitializeWeights(network);
		trainer.Train(network, training_set);

		// then
		network.ComputeOutput(training_set.front().first);
		EXPECT_LT(network.GetOutputActivation(0), 0.1);

		network.ComputeOutput(training_set.back().first);
		EXPECT_LT(network.GetOutputActivation(0), 0.1);

		network.ComputeOutput(training_set.at(1).first);
		EXPECT_GT(network.GetOutputActivation(0), 0.9);

		network.ComputeOutput(training_set.at(2).first);
		EXPECT_GT(network.GetOutputActivation(0), 0.9);
	}
}