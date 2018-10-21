#include "pch.h"
#include "TestFixtures.h"

namespace NNSLibTest
{
	using namespace NNS::Models;
	using namespace NNS::Training;
	using namespace NNS::Activation;

	TEST(TrainingErrorStateTests, ComputeEpochGradientForPredefinedWeights)
	{
		// given
		auto e = 10'000'000;
		auto training_set = testHelpers::ReadTrainingDataSet("C:\\Repos\\NNSimulator\\NnsLib.Tests\\TestData\\xor_i2_o1.txt");
		auto network = GetMultilayerPerceptronWithPredefinedWeights();
		TrainingErrorState errorState(*network, training_set);
		
		// when
		const auto error = errorState.ComputeEpochGradient();

		// then
		auto gradient = errorState.GetErrorGradient();
		EXPECT_EQ(0.0118771, trunc(error*e) / e);
		EXPECT_EQ(0.0052554, trunc(gradient[0][0][0]*e) / e);
		EXPECT_EQ(-0.0185337, trunc(gradient[0][0][1]*e) / e);
		EXPECT_EQ(-0.0163232, trunc(gradient[0][0][2]*e) / e);
		EXPECT_EQ(-0.0016917, trunc(gradient[0][1][0]*e) / e);
		EXPECT_EQ(-0.0129071, trunc(gradient[0][1][1]*e) / e);
		EXPECT_EQ(-0.0172947, trunc(gradient[0][1][2]*e) / e);
		EXPECT_EQ(0.0102798, trunc(gradient[1][0][0]*e) / e);
		EXPECT_EQ(0.0231979, trunc(gradient[1][0][1]*e) / e);
		EXPECT_EQ(-0.0084707, trunc(gradient[1][0][2]*e) / e);
	}
}