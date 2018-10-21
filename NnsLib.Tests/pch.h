//
// pch.h
// Header for standard system include files.
//

#pragma once

#include "gtest/gtest.h"

#include <cmath>
#include <memory>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include <Models/MultilayerPerceptron.h>
#include <Models/KohonenNetwork.h>
#include <Initialization/RandomWeightInitializer.h>
#include <Optimization/Backpropagation.h>
#include <Optimization/ConjugateGradient.h>
#include <Optimization/SimulatedAnnealing.h>
#include <Training/SupervisedTraining.h>

#include "HelperFunctions.h"
