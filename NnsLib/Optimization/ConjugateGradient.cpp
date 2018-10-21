#include "pch.h"
#include "Optimization/ConjugateGradient.h"
#include <ctime>

namespace NNS 
{
	namespace Optimization 
	{
		ConjugateGradient::ConjugateGradient(ErrorUnit errorDeltaTolerance,
			size_t maxInternalIter, int maxRandomRetry):
			errorDeltaTolerance{ errorDeltaTolerance }, maxInternalIterations{ maxInternalIter }, maxRandomRetry{ maxRandomRetry }
		{
			rngEngine.seed(static_cast<int>(time(0)));
		}

		void ConjugateGradient::Free() const
		{
			delete this;
		}

		void ConjugateGradient::Initialize(IFeedforwardNetwork& network)
		{
			tempMatrixG.clear();
			searchDirectionH.clear();

			networkmap = network.GetNetworkLayerMap();
		}

		bool ConjugateGradient::OptimizeWeights(IFeedforwardNetwork& network, TrainingErrorState& errorState)
		{
			/* Initialize matrices used as a sequence of work vectors and search directions. */
			tempMatrixG = errorState.GetErrorGradient();
			searchDirectionH = errorState.GetErrorGradient();

			/* Obtain previous objective function value */
			auto previous_error = errorState.GetErrorVector().back();

			/* Conjugate Gradient max iteration protection. */
			for (size_t iter = 0; iter < maxInternalIterations; ++iter)
			{

				/* Check absolute error for convergence. */
				auto error = LineMinimization(network, errorState, previous_error, 10, 1.0e-10, 0.5);

				if (error < 0.0) /* Forced end of calculation. */
				{
					errorState.UpdateErrorVector(error);
					return true;
				}

				/* Check the relative error for convergence. */
				/* If the directional minimization is not effective then try some random directions. */
				/* Thats an insurance policy against getting stuck at a saddle point. */
				if ((2.0 * (previous_error - error)) <= (errorDeltaTolerance * (previous_error + error + 1.e-10)))
				{
					previous_error = error; /* But first exhaust weight gradient. */
					error = errorState.ComputeEpochGradient(); /* Recompute gradient. */
					error = LineMinimization(network, errorState, error, 15, 1.0e-10, 1.e-3);

					int retry;
					for (retry = 0; retry < maxRandomRetry; ++retry)
					{
						for (size_t i = 1; i < networkmap.size(); ++i) /* For each layer ( minus input layer ). */
							for (size_t j = 0; j < networkmap[i]; ++j) /* For each neuron. */
								for (size_t k = 0; k < networkmap[i - 1] + 1; ++k) /* For each connection + bias. */
									errorState.GetErrorGradient()[i - 1][j][k] = (0.5 - rngUni01(rngEngine)) / 10;

						error = LineMinimization(network, errorState, error, 10, 1.e-10, 1.e-2);
						if (error < 0.0) /* Forced end of calculation. */
						{
							errorState.UpdateErrorVector(error);
							return true;
						}

						if (retry < maxRandomRetry / 2)
							continue;

						if ((2.0 * (previous_error - error)) > (errorDeltaTolerance * (previous_error + error + 1.e-10)))
							break;   /* Get out of retry loop if we improved enough */
					} /* End of For loop */

					if (retry == maxRandomRetry) /* If we exhausted all tries, its probably hopeless. */
					{
						//_epochErrorVector.push_back( error );
						break;
					}

					tempMatrixG = errorState.GetErrorGradient();
					searchDirectionH = errorState.GetErrorGradient();
				} // End of "If this direction give poor result"

				previous_error = error;
				//_epochErrorVector.push_back( error );

				/* Setup for next iteration. */
				error = errorState.ComputeEpochGradient(); /* Recompute gradient. */

				/* Calculate gamma constant. */
				auto gamma = ComputeGamma(errorState, tempMatrixG);
				if (gamma < 0.0) /* Restricting gamma to range <0; 1> almost always speeds convergence for neural network learning. */
					gamma = 0.0;
				if (gamma > 1.0)
					gamma = 1.0;

				/* Use gamma constant along with work matrix 'g' and search direction 'h' to find the search direction for the next iteration. */
				ComputeNewSearchDirection(errorState, gamma, tempMatrixG, searchDirectionH);
			}

			return false; /* set to TRUE will bypass SupervisedTraining::Train() loop and execute this method only once. */
		}

		ErrorUnit ConjugateGradient::ComputeGamma(TrainingErrorState& errorState, ErrorGradientMatrix& tempMatrixG)
		{
			ErrorUnit denominator{};
			ErrorUnit numerator{};

			for (size_t i = 1; i < networkmap.size(); ++i)
			{ /* For each layer ( minus input layer ). */
				for (size_t j = 0; j < networkmap[i]; ++j)
				{ /* For each neuron. */
					for (size_t k = 0; k < networkmap[i - 1] + 1; ++k)
					{ /* For each connection + bias. */
						denominator += pow(tempMatrixG[i - 1][j][k], 2.0);
						numerator += (errorState.GetErrorGradient()[i - 1][j][k] - tempMatrixG[i - 1][j][k]) * errorState.GetErrorGradient()[i - 1][j][k]; /* error gradient is negative gradient */
					}
				}
			}

			if (denominator == 0) /* Should never happen (means gradient is zero!) */
				return {};
			else
				return numerator / denominator;
		}


		void ConjugateGradient::ComputeNewSearchDirection(TrainingErrorState& errorState, ErrorUnit gamma, ErrorGradientMatrix& tempMatrixG, DirectionMatrix& searchDirectionH)
		{
			//tempMatrixG = errorGradient;
			for (size_t i = 1; i < static_cast<int>(networkmap.size()); ++i)
			{ /* For each layer ( minus input layer ). */
				for (size_t j = 0; j < networkmap[i]; ++j)
				{ /* For each neuron. */
					for (size_t k = 0; k < networkmap[i - 1] + 1; ++k)
					{ /* For each connection + bias. */
						tempMatrixG[i - 1][j][k] = errorState.GetErrorGradient()[i - 1][j][k]; /* Save previous directon. */
						searchDirectionH[i - 1][j][k] = tempMatrixG[i - 1][j][k] + gamma * searchDirectionH[i - 1][j][k];
						errorState.GetErrorGradient()[i - 1][j][k] = searchDirectionH[i - 1][j][k];
					}
				}
			}
			//_errorGradient = searchDirectionH;
		}

		ErrorUnit ConjugateGradient::LineMinimization(IFeedforwardNetwork& network, TrainingErrorState& errorState, ErrorUnit startError, size_t maxIterations, ErrorUnit epsilon, ErrorUnit tolerance)
		{
			ErrorUnit step /* next step */, max_step, x1, x2, x3, t1 /* temporal x1 */, t2 /* temporal x2 */, numerator, denominator /* for parabolic fit */;
			ErrorUnit current_error /* x2 error */, error /* x3 error */, previous_error /* x1 error */, step_error /* temporal error */;

			ErrorUnit  first_step = 2.5; /* Heuristically found best. */

			WeightMatrix baseWeights = network.GetWeightMatrix(); /* Establishes a baseWeights for stepping out ( saves the weights, so they serve as X0 ). */

			StepOut(network, first_step, errorState.GetErrorGradient()/* direction Xd */, baseWeights); /* Take one step out in the gradient direction. Computes new weights appropriately. */
			error = errorState.ComputeEpochError(); /* Compute epoch error. */

			if (error > startError) /* If the error increased, we may have stepped too far. reverse the role of the two points. */
			{
				ReverseDirection(errorState.GetErrorGradient()); /* Negate the direction */
				x1 = -first_step; /* Use -1, 0 and 1.618 as first three steps. */
				x2 = 0.0;

				previous_error = error;
				current_error = startError;
			}
			else /* Otherwise use 0, 1 and 2.618 as first three steps. */
			{
				x1 = 0.0;
				x2 = first_step;

				previous_error = startError;
				current_error = error;
			}

			/*if (isTrainingAborted)
				return -current_error;*/

			/* At this point we have taken a single step and the function decreased. */
			/* Take one more ( 3rd ) step in the golden ratio. */
			x3 = x2 + 1.618034 * first_step;
			StepOut(network, x3, errorState.GetErrorGradient(), baseWeights);
			error = errorState.ComputeEpochError();

			/*
			We now have three points x1, x2 and x3 with corresponding errors of 'previous_error', 'current_error' and 'error'.
			Endlessly loop until we bracket the minimum with the outer two.
			*/

			while (error < current_error) /* As long as we are descending. */
			{
				/*
				Try a parabolic fit to estimate the location of the minimum.
				*/

				t1 = (x2 - x1) * (current_error - error);
				t2 = (x2 - x3) * (current_error - previous_error);
				denominator = 2.0 * (t2 - t1);

				if (fabs(denominator) < epsilon)
				{
					if (denominator > 0.0)
						denominator = epsilon;
					else
						denominator = -epsilon;
				}

				step = x2 + ((x2 - x1) * t1 - (x2 - x3) * t2) / denominator; /* Here if perfect. */
				//step = x2 - ( ( x2 - x1 ) * t1 - ( x2 - x3 ) * t2 ) / denominator;
				max_step = x2 + 200.0 * (x3 - x2); /* Don't jump too far */

				if ((x2 - step) * (step - x3) > 0.0) /* It's between x2 and x3. */
				{
					StepOut(network, step, errorState.GetErrorGradient(), baseWeights);
					step_error = errorState.ComputeEpochError();

					if (step_error < error) /* It worked!  We found min between x2 and x3. */
					{
						x1 = x2;
						x2 = step;
						previous_error = current_error;
						current_error = step_error;
						break;
					}
					else if (step_error > current_error) /* Slight miscalc.  Min at x2. */
					{
						x3 = step;
						error = step_error;
						break;
					}
					else /* Parabolic fit was total waste of time.  Use default. */
					{
						step = x3 + 1.618034 * (x3 - x2);
						StepOut(network, step, errorState.GetErrorGradient(), baseWeights);
						step_error = errorState.ComputeEpochError();
					}
				}
				else if ((x3 - step) * (step - max_step) > 0.0) /* Between x3 and lim. */
				{
					StepOut(network, step, errorState.GetErrorGradient(), baseWeights);
					step_error = errorState.ComputeEpochError();
					if (step_error < error)  /* Decreased, so advance by golden ratio. */
					{
						x2 = x3;
						x3 = step;
						step = x3 + 1.618034 * (x3 - x2);
						current_error = error;
						error = step_error;
						StepOut(network, step, errorState.GetErrorGradient(), baseWeights);
						step_error = errorState.ComputeEpochError();
					}
				}
				else if ((step - max_step) * (max_step - x3) >= 0.0) /* Beyond limit. */
				{
					step = max_step;
					StepOut(network, step, errorState.GetErrorGradient(), baseWeights);
					step_error = errorState.ComputeEpochError();
					if (step_error < error) {  /* Decreased, so advance by golden ratio. */
						x2 = x3;
						x3 = step;
						step = x3 + 1.618034 * (x3 - x2);
						current_error = error;
						error = step_error;
						StepOut(network, step, errorState.GetErrorGradient(), baseWeights);
						step_error = errorState.ComputeEpochError();
					}
				}
				else  /* Wild!  Reject parabolic and use golden ratio. */
				{
					step = x3 + 1.618034 * (x3 - x2);
					StepOut(network, step, errorState.GetErrorGradient(), baseWeights);
					step_error = errorState.ComputeEpochError();
				}

				/* Shift three points and continue endless loop. */

				x1 = x2;
				x2 = x3;
				x3 = step;
				previous_error = current_error;
				current_error = error;
				error = step_error;
			} /* End of While loop. */

			StepOut(network, x2, errorState.GetErrorGradient(), baseWeights); /* Leave weights at minimum */

			if (x1 > x3)  /* We may have switched direction at start. */
			{
				t1 = x1;    /* Brent's method, which follows, assumes ordered parameter. */
				x1 = x3;
				x3 = t1;
			}

			//if (isTrainingAborted) /* If forced end of calculation. */
			//{
			//	UpdateDirection(x2, errorGradient);/* Make it be the actual dist moved (multiplies search direction vector by the specified value of the parameter t,
			//								  so that it reflects the actual vector difference between the point corresponding to t and the baseWeights point X0. */
			//	return -current_error;
			//}

			/* 2nd Step starts here. */
			/* At this point we have bounded the minimum between x1 and x3. */
			/* Go to the refinement stage. We use Brent's algorithm. */

			ErrorUnit xlow, xhigh, xbest, testdist; /* Declare variables. */
			ErrorUnit prevdist, frecent, fthirdbest, fsecbest, fbest;
			ErrorUnit tol1, tol2, xrecent, xthirdbest, xsecbest, xmid;

			prevdist = 0.0; /* Initialize prevdist. */
			step = 0.0; /* Zero step value. */

			xbest = xsecbest = xthirdbest = x2; /* xbest has the min function so far (or latest if tie). */
			xlow = x1; /* We always keep the minimum bracketed between xlow and xhigh. */
			xhigh = x3;

			fbest = fsecbest = fthirdbest = current_error;

			for (size_t i = 0; i < maxIterations; i++) /* Loop with limit of iterations */
			{

				xmid = 0.5 * (xlow + xhigh);
				tol1 = tolerance * (fabs(xbest) + epsilon);
				tol2 = 2.0 * tol1;

				/* The following convergence test simultaneously makes sure xhigh and xlow are close relative to tol2, and that xbest is near the midpoint. */
				if (fabs(xbest - xmid) <= (tol2 - 0.5 * (xhigh - xlow)))
					break;

				if (fabs(prevdist) > tol1) /* If we moved far enough try parabolic fit. */
				{
					t1 = (xbest - xsecbest) * (fbest - fthirdbest); /* Temps for the parabolic estimate */
					t2 = (xbest - xthirdbest) * (fbest - fsecbest);
					numerator = (xbest - xthirdbest) * t2 - (xbest - xsecbest) * t1;
					denominator = 2.0 * (t1 - t2);  /* Estimate will be numerator / denominator */
					testdist = prevdist;  /* Will soon verify interval is shrinking. */
					prevdist = step; /* Save for next iteration. */

					if (denominator != 0.0) /* Avoid dividing by zero. */
						step = numerator / denominator; /* This is the parabolic estimate to min */
					else
						step = 1.e30; /* Assures failure of next test */

					if ((fabs(step) < fabs(0.5 * testdist)) /* If shrinking */
						&& (step + xbest > xlow) /* and within known bounds */
						&& (step + xbest < xhigh))
					{
						xrecent = xbest + step; /* Then we can use the parabolic estimate */
						if ((xrecent - xlow < tol2) || (xhigh - xrecent < tol2)) /* If we are very close to known bounds */
						{
							if (xbest < xmid) /* Then stabilize */
								step = tol1;
							else
								step = -tol1;
						}
					}
					else /* Parabolic estimate poor, so use golden section. */
					{
						if (xbest >= xmid)
							prevdist = xlow - xbest;
						else
							prevdist = xhigh - xbest;
						step = 0.3819660 * prevdist;
					}
				}
				else /* prevdist did not exceed tol1: we did not move far enough to justify a parabolic fit.  Use golden section. */
				{
					if (xbest >= xmid)
						prevdist = xlow - xbest;
					else
						prevdist = xhigh - xbest;
					step = 0.3819660 * prevdist;
				}

				if (fabs(step) >= tol1) /* In order to numeratorically justify another trial we must move a decent distance. */
				{
					xrecent = xbest + step;
				}
				else
				{
					if (step > 0.0)
						xrecent = xbest + tol1;
					else
						xrecent = xbest - tol1;
				}

				/* At long last we have a trial point 'xrecent'.  Evaluate the function. */

				StepOut(network, xrecent, errorState.GetErrorGradient(), baseWeights);
				frecent = errorState.ComputeEpochError();

				if (frecent <= fbest) /* If we improved... */
				{
					if (xrecent >= xbest) /* Shrink the (xlow,xhigh) interval by replacing the appropriate endpoint. */
						xlow = xbest;
					else
						xhigh = xbest;

					xthirdbest = xsecbest; /* Update x and f values for best, second and third best. */
					xsecbest = xbest;
					xbest = xrecent;
					fthirdbest = fsecbest;
					fsecbest = fbest;
					fbest = frecent;
				}
				else /* We did not improve */
				{
					if (xrecent < xbest) /* Shrink the ( xlow; xhigh ) interval by replacing the appropriate endpoint. */
						xlow = xrecent;
					else
						xhigh = xrecent;

					if ((frecent <= fsecbest) || (xsecbest == xbest)) /* If we at least beat the second best or we had a duplication. */
					{
						xthirdbest = xsecbest;  /* We can update the second and third best, though not the best.*/
						xsecbest = xrecent;
						fthirdbest = fsecbest;  /* Recall that we started iters with best, sec and third all equal. */
						fsecbest = frecent;
					}
					else if ((frecent <= fthirdbest) /* Maybe at least we can beat the third best  */
						|| (xthirdbest == xbest) /* or rid ourselves of a duplication (which is how we start the iterations) */
						|| (xthirdbest == xsecbest))
					{
						xthirdbest = xrecent;
						fthirdbest = frecent;
					}
				}
			} /* End of For loop */

			StepOut(network, xbest, errorState.GetErrorGradient(), baseWeights); /* Leave coefficients at minimum */
			UpdateDirection(xbest, errorState.GetErrorGradient()); /* Make it be the actual distance moved. */

			//if (isTrainingAborted) /* If forced end of calculation. */
			//	return -fbest;
			//else
				return fbest;
		}

		void ConjugateGradient::StepOut(IFeedforwardNetwork& network, ErrorUnit step, DirectionMatrix& direction, WeightMatrix& baseWeights)
		{
			for (size_t i = 1; i < networkmap.size(); ++i)
			{	/* For each layer ( minus input layer ). */
				for (size_t j = 0; j < networkmap[i]; ++j)
				{	/* For each neuron. */
					for (size_t k = 0; k < networkmap[i - 1] + 1; ++k)
					{	/* For each connection + bias. */
						network.Weight(i, j, k) = baseWeights[i - 1][j][k] + step * direction[i - 1][j][k];
					}
				}
			}
		}


		void ConjugateGradient::UpdateDirection(ErrorUnit step, DirectionMatrix& direction)
		{
			for (size_t i = 1; i < networkmap.size(); ++i)
			{	/* For each layer ( minus input layer ). */
				for (size_t j = 0; j < networkmap[i]; ++j)
				{	/* For each neuron. */
					for (size_t k = 0; k < networkmap[i - 1] + 1; ++k)
					{	/* For each connection + bias. */
						direction[i - 1][j][k] *= step;
					}
				}
			}
		}

		void ConjugateGradient::ReverseDirection(DirectionMatrix& direction)
		{
			for (size_t i = 1; i < networkmap.size(); ++i)
			{	/* For each layer ( minus input layer ). */
				for (size_t j = 0; j < networkmap[i]; ++j)
				{	/* For each neuron. */
					for (size_t k = 0; k < networkmap[i - 1] + 1; ++k)
					{	/* For each connection + bias. */
						direction[i - 1][j][k] = -direction[i - 1][j][k];
					}
				}
			}
		}
	}
}