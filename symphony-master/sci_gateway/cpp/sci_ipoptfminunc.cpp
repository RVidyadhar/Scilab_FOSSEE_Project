// Copyright (C) 2015 - IIT Bombay - FOSSEE
//
// Author: R.Vidyadhar & Vignesh Kannan
// Organization: FOSSEE, IIT Bombay
// Email: rvidhyadar@gmail.com & vignesh2496@gmail.com
// This file must be used under the terms of the CeCILL.
// This source file is licensed as described in the file COPYING, which
// you should have received as part of this distribution.  The terms
// are also available at
// http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt


#include "sci_iofunc.hpp"
#include "IpIpoptApplication.hpp"
#include "minuncNLP.hpp"

extern "C"
{
#include <api_scilab.h>
#include <Scierror.h>
#include <BOOL.h>
#include <localization.h>
#include <sciprint.h>
#include <iostream>

using namespace std;

int sci_solveminuncp(char *fname)
{
	using namespace Ipopt;

	CheckInputArgument(pvApiCtx, 5, 7);
	CheckOutputArgument(pvApiCtx, 6, 6);
	
	// Error management variable
	SciErr sciErr;

	//Function pointers, input matrix(Starting point) pointer, flag variable 
	int* funptr=NULL;
	int* gradhesptr=NULL;
	double* x0ptr=NULL;
	double flag;
        

        // Input arguments
	double *cpu_time=NULL,*max_iter=NULL;
	static unsigned int nVars = 0,nCons = 0;
	unsigned int temp1 = 0,temp2 = 0, iret = 0;
	int x0_rows, x0_cols;
	
	// Output arguments
	double *fX = NULL, ObjVal=0,iteration=0;
	double *fGrad=  NULL;
	double *fHess=  NULL;
	int rstatus = 0;

	////////// Manage the input argument //////////
	
	//Objective Function
	if(getFunctionFromScilab(1,&funptr))
	{
		return 1;
	}

 	//Function for gradient and hessian
	if(getFunctionFromScilab(2,&gradhesptr))
	{
		return 1;
	}

	//x0(starting point) matrix from scilab
	if(getDoubleMatrixFromScilab(3, &x0_rows, &x0_cols, &x0ptr))
	{
		return 1;
	}

       
        //Getting number of iterations
        if(getFixedSizeDoubleMatrixInList(4,2,temp1,temp2,&max_iter))
	{
		return 1;
	}

	//Getting Cpu Time
	if(getFixedSizeDoubleMatrixInList(4,4,temp1,temp2,&cpu_time))
	{
		return 1;
	}

	if(getDoubleFromScilab(5, &flag))
	{
		return 1;
	}

        //Initialization of parameters
	nVars=x0_cols;
	nCons=0;
        
        // Starting Ipopt

	SmartPtr<minuncNLP> Prob = new minuncNLP(nVars, nCons, x0ptr, flag);
	SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
	app->RethrowNonIpoptException(true);

	////////// Managing the parameters //////////

	app->Options()->SetNumericValue("tol", 1e-7);
	app->Options()->SetIntegerValue("max_iter", (int)*max_iter);
	app->Options()->SetNumericValue("max_cpu_time", *cpu_time);

	///////// Initialize the IpoptApplication and process the options /////////
	ApplicationReturnStatus status;
 	status = app->Initialize();
	if (status != Solve_Succeeded) {
	  	sciprint("\n*** Error during initialization!\n");
   	 return (int) status;
 	 }
	 // Ask Ipopt to solve the problem
	
	 status = app->OptimizeTNLP(Prob);

	 rstatus = Prob->returnStatus();
         

	////////// Manage the output argument //////////

	if (rstatus == 0 | rstatus == 1 | rstatus == 2)
	{
		fX = Prob->getX();
		fGrad = Prob->getGrad();
		fHess = Prob->getHess();
		ObjVal = Prob->getObjVal();
		iteration = Prob->iterCount();

		if (returnDoubleMatrixToScilab(1, 1, nVars, fX))
		{
			return 1;
		}

		if (returnDoubleMatrixToScilab(2, 1, 1, &ObjVal))
		{
			return 1;
		}

		if (returnIntegerMatrixToScilab(3, 1, 1, &rstatus))
		{
			return 1;
		}

		if (returnDoubleMatrixToScilab(4, 1, 1, &iteration))
		{
			return 1;
		}
		
		if (returnDoubleMatrixToScilab(5, 1, nVars, fGrad))
		{
			return 1;
		}
		if (returnDoubleMatrixToScilab(6, 1, nVars*nVars, fHess))
		{
			return 1;
		}
	}

	else
	{
		if (returnDoubleMatrixToScilab(1, 0, 0, fX))
		{
			return 1;
		}

		if (returnDoubleMatrixToScilab(2, 1, 1, &ObjVal))
		{
			return 1;
		}

		if (returnIntegerMatrixToScilab(3, 1, 1, &rstatus))
		{
			return 1;
		}

		if (returnDoubleMatrixToScilab(4, 1, 1, &iteration))
		{
			return 1;
		}
		if (returnDoubleMatrixToScilab(5, 1, nVars, fGrad))
		{
			return 1;
		}
		if (returnDoubleMatrixToScilab(6, 1, nVars*nVars, fHess))
		{
			return 1;
		}
	}

	// As the SmartPtrs go out of scope, the reference count
	// will be decremented and the objects will automatically
	// be deleted.*/

	return 0;
}
}
