#include "pch.h"

namespace testHelpers
{
	TrainingDataSet ReadTrainingDataSet(const string& filePath)
	{
		TrainingDataSet training_set;

		ifstream infile(filePath);

		string s;
		while (getline(infile, s))
		{
			istringstream ss(s);
			vector<SignalUnit> records;

			while (ss)
			{
				string s;
				if (!getline(ss, s, '\t'))
					break;
				records.push_back(atof(s.c_str()));
			}

			ActivationVector inputSet = Eigen::Map<ActivationVector>(records.data(), records.size()-1);
			ActivationVector outputSet(1); outputSet << records.back();

			training_set.emplace_back(inputSet, outputSet);
		}

		return training_set;
	}

	// Returns time stamp counter
	long long ReadTSC() 
	{		
		int dummy[4];			// For unused returns
		volatile int DontSkip;	// Volatile to prevent optimizing
		long long clock;		// Time
		__cpuid(dummy, 0);		// Serialize
		DontSkip = dummy[0];	// Prevent optimizing away cpuid
		clock = __rdtsc();		// Read time
		return clock;
	}

}