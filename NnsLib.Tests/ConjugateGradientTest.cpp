#include "pch.h"

namespace NNSLibTest
{
	using std::vector;
	using std::shared_ptr;
	using std::unique_ptr;

	using namespace NNS::Models;
	using namespace NNS::Training;
	using namespace NNS::Activation;
	using namespace NNS::Initialization;
	using namespace NNS::Optimization;

	TEST(ConjugateGradientTest, ConjugateGradient_Xor2to1Problem)
	{
		// given
		MultilayerPerceptron network{ 2, 3, 1 };
		RandomWeightInitializer weight_init{ 0.5f };
		ConjugateGradient algorithm{ 0.0001f, 1000, 5 };
		SupervisedTraining trainer{ algorithm, 1000, 0.00001f };
		TrainingDataSet training_set = testHelpers::ReadTrainingDataSet("C:\\Repos\\NNSimulator\\NnsLib.Tests\\TestData\\xor_i2_o1.txt");

		// when
		weight_init.InitializeWeights(network);
		trainer.Train(network, training_set);

		// then
		network.ComputeOutput(training_set.front().first);
		EXPECT_LE(network.GetOutputActivation(0), 0.1);

		network.ComputeOutput(training_set.back().first);
		EXPECT_LE(network.GetOutputActivation(0), 0.1);

		network.ComputeOutput(training_set.at(1).first);
		EXPECT_GE(network.GetOutputActivation(0), 0.9);

		network.ComputeOutput(training_set.at(2).first);
		EXPECT_GE(network.GetOutputActivation(0), 0.9);
	}

	TEST(ConjugateGradientTest, Xor3to1Problem_WithSimulatedAnnealing)
	{
		// given
		ErrorUnit errorThreshold = 0.00001f;

		MultilayerPerceptron network{ 3, 3, 5, 1 };
		RandomWeightInitializer weight_init{ 0.5 };
		ConjugateGradient algorithm{ 0.0001f, 1000, 5 };
		SupervisedTraining trainer{ algorithm, 1000, errorThreshold };
		TrainingDataSet training_set = testHelpers::ReadTrainingDataSet("C:\\Repos\\NNSimulator\\NnsLib.Tests\\TestData\\xor_i3_o1.txt");
		SimulatedAnnealing elm{ SimulatedAnnealingConfig{ 1.0f, 0.01f, errorThreshold, 5, 100, 30, RandomDistributionMethod::Normal, 0.5f } };
		trainer.SetEludingLocalMinimaMethod(&elm);

		// when
		weight_init.InitializeWeights(network);
		trainer.Train(network, training_set);

		// then
		network.ComputeOutput(training_set.front().first);
		EXPECT_LE(network.GetOutputActivation(0), 0.1);

		network.ComputeOutput(training_set.back().first);
		EXPECT_GE(network.GetOutputActivation(0), 0.9);

		network.ComputeOutput(training_set.at(2).first);
		EXPECT_GE(network.GetOutputActivation(0), 0.9);

		network.ComputeOutput(training_set.at(4).first);
		EXPECT_GE(network.GetOutputActivation(0), 0.9);
	}

}