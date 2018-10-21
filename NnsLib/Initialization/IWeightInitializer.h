#pragma once

#include <memory>

#include "Common/IBase.h"
#include "Models/IFeedforwardNetwork.h"

namespace NNS 
{
	namespace Initialization 
	{
		using NNS::Models::IFeedforwardNetwork;

		class IWeightInitializer : public IBase {
		public:
			using Ptr = std::unique_ptr<IWeightInitializer, SDeleter>;
		public:
			virtual void InitializeWeights(IFeedforwardNetwork& network) = 0;
		};
	} 
}