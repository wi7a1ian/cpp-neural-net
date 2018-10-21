#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <functional>

#include "Types/Units.h"

#ifndef M_PI
#	define M_PI       3.14159265358979323846
#endif // !M_PI

namespace NNS 
{
	namespace Activation 
	{
		using NNS::Types::SignalUnit;
		using ActivationFunctionPtr = std::function<SignalUnit(SignalUnit x)> ;

		template<typename T = SignalUnit>
		struct TresholdActivationFunction
		{
			T operator()(T x) const
			{
				return (x > 0.0) ? 1.0 : 0.0;
			}

			T Deriv(T x) const
			{
				return {};
			}
		};

		template<typename T = SignalUnit>
		struct LogisticActivationFunction
		{
			T operator()(T x) const
			{
				return (1 / (1 + exp(-x)));
			}

			T Deriv(T x) const
			{
				return (*this)(x) * (1 - (*this)(x));
			}
		};


		template<typename T = SignalUnit>
		struct HiperbolicTangensActivationFunction
		{
			T operator()(T x) const
			{
				return tanh(x);
			}

			T Deriv(T x) const
			{
				return (1.0 - pow(tanh(x), 2.0));
			}
		};

		template<typename T = SignalUnit>
		struct Kenue1ActivationFunction
		{
			T operator()(T x) const
			{
				return (2.0 / M_PI) * atan(sinh(x));
			}

			T Deriv(T x) const
			{
				return (2.0 / M_PI) * pow(cosh(x), -1.0);
			}
		};

		template<typename T = SignalUnit>
		struct Kenue2ActivationFunction
		{
			T operator()(T x) const
			{
				return (2.0 / M_PI) * (tanh(x) / cosh(x) + atan(sinh(x)));
			}

			T Deriv(T x) const
			{
				return (4.0 / M_PI) * pow(pow(cosh(x), -1.0), 3.0);
			}
		};
	}
}