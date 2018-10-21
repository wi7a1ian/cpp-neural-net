#pragma once

#include <intrin.h>
using std::ifstream;
using std::string;
using std::istringstream;

using namespace NNS::Training;

namespace testHelpers
{
	TrainingDataSet ReadTrainingDataSet(const string& filePath);
	long long ReadTSC();
}