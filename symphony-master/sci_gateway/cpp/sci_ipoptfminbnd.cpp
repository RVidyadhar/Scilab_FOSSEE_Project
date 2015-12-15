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
#include "minbndNLP.hpp"

extern "C"
{
#include <api_scilab.h>
#include <Scierror.h>
#include <BOOL.h>
#include <localization.h>
#include <sciprint.h>
#include <iostream>

using namespace std;

int sci_solveminbndp(char *fname)
{
	using namespace Ipopt;

	CheckInputArgument(pvApiCtx, 5, 5); 
	CheckOutputArgument(pvApiCtx, 4, 4);
	
	// Error management variable
	SciErr sciErr;

	//Function pointers,lower bound and upper bound pointers 
	int* funptr=NULL;
	int* gradhesptr=NULL;
	double* varLB=NULL;
	double* varUB=NULL;

        // Input arguments
	double *cpu_time=NULL,*max_iter=NULL,*tol_val=NULL;
	static unsigned int nVars = 0,nCons = 0;
	unsigned int temp1 = 0,temp2 = 0, iret = 0;
	int x1_rows, x1_cols, x2_rows, x2_cols;
	
	// Output arguments
	double *fX = NULL, ObjVal=0,iteration=0;
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

	//x1(lower bound) matrix from scilab
	if(getDoubleMatrixFromScilab(3, &x1_rows, &x1_cols, &varLB))
	{
		return 1;
	}
     
	//x2(upper bound) matrix from scilab
	if(getDoubleMatrixFromScilab(4, &x2_rows, &x2_cols, &varUB))
	{
		return 1;
	}

        //Getting number of iterations
        if(getFixedSizeDoubleMatrixInList(5,2,temp1,temp2,&max_iter))
	{
		return 1;
	}

	//Getting Cpu Time
	if(getFixedSizeDoubleMatrixInList(5,4,temp1,temp2,&cpu_time))
	{
		return 1;
	}

	//Getting Tolerance Value
	if(getFixedSizeDoubleMatrixInList(5,6,temp1,temp2,&tol_val))
	{
		return 1;
	}
 

        //Initialization of parameters
	nVars=1;
	nCons=0;
        
        // Starting Ipopt

	SmartPtr<minbndNLP> Prob = new minbndNLP(nVars,nCons,varLB,varUB);
	
	SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
	app->RethrowNonIpoptException(true);

	////////// Managing the parameters //////////

	app->Options()->SetNumericValue("tol", *tol_val);
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
	}

	return 0;
}
}
