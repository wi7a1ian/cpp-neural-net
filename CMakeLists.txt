cmake_minimum_required (VERSION 2.6)
project (NnsLib)

set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/NnsLib)
set(SOURCES 
	${SRC}/Common/ActivationFunctions.h
	${SRC}/Common/IBase.h
	${SRC}/Common/InterfaceHelpers.h
	${SRC}/Initialization/IWeightInitializer.h
	${SRC}/Initialization/RandomWeightInitializer.h
	${SRC}/Models/IFeedforwardNetwork.h
	${SRC}/Models/KohonenNetwork.h
	${SRC}/Models/MultilayerPerceptron.h
	${SRC}/Models/FeedforwardNetworkBase.h
	${SRC}/Optimization/IWeightOptimizer.h
	${SRC}/Optimization/SimulatedAnnealing.h
	${SRC}/Optimization/Backpropagation.h
	${SRC}/Optimization/ConjugateGradient.h
	${SRC}/pch.h
	${SRC}/Training/ITrainingAlgorithm.h
	${SRC}/Training/SupervisedTraining.h
	${SRC}/Training/TrainingErrorState.h
	${SRC}/Types/Collections.h
	${SRC}/Types/Units.h
	${SRC}/Initialization/RandomWeightInitializer.cpp
	${SRC}/Models/KohonenNetwork.cpp
	${SRC}/Models/MultilayerPerceptron.cpp
	${SRC}/Models/FeedforwardNetworkBase.cpp
	${SRC}/Optimization/SimulatedAnnealing.cpp
	${SRC}/Optimization/Backpropagation.cpp
	${SRC}/Optimization/ConjugateGradient.cpp
	${SRC}/Training/SupervisedTraining.cpp
	${SRC}/Training/TrainingErrorState.cpp	
    ${SRC}/pch.cpp)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/eigen")
include_directories("${SRC}")

add_library(NnsLib ${SOURCES})

if(MSVC)
  target_compile_options(NnsLib PRIVATE /W4)
else()
  target_compile_options(NnsLib PRIVATE
                         -Wall
                         -Wextra # reasonable and standard
                         -Wshadow # warn the user if a variable declaration shadows one from a
                                  # parent context
#                         -Wnon-virtual-dtor # warn the user if a class with virtual functions has a
#                                            # non-virtual destructor. This helps catch hard to
#                                            # track down memory errors
#                         -Wold-style-cast # warn for c-style casts
#                         -Wcast-align # warn for potential performance problem casts
#                         -Wunused # warn on anything being unused
#                         -Woverloaded-virtual # warn if you overload (not override) a virtual
#                                              # function
#                         -Wpedantic # warn if non-standard C++ is used
#                         -Wconversion # warn on type conversions that may lose data
#                         -Wsign-conversion # warn on sign conversions
#                         -Wmisleading-indentation # warn if identation implies blocks where blocks
#                                                  # do not exist
#                         -Wduplicated-cond # warn if if / else chain has duplicated conditions
#                         -Wduplicated-branches # warn if if / else branches have duplicated code
#                         -Wlogical-op # warn about logical operations being used where bitwise were
#                                      # probably wanted
#                         -Wnull-dereference # warn if a null dereference is detected
#                         -Wuseless-cast # warn if you perform a cast to the same type
#                         -Wdouble-promotion # warn if float is implicit promoted to double
#                         -Wformat=2 # warn on security issues around functions that format output
                         )
endif()

target_compile_features(NnsLib PRIVATE cxx_std_17)

