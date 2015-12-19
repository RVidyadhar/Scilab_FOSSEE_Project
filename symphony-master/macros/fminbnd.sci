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


function [xopt,fopt,exitflag,output] = fminbnd (varargin)
  // Solves a minimum of single-variable function on open bounded interval
  //
  //   Calling Sequence
  //   xopt = fminunc(_f,x1,x2)
  //   xopt = fminunc(_f,x1,x2,options)
  //   [xopt,fopt] = fminunc(.....)
  //   [xopt,fopt,exitflag]= fminunc(.....)
  //   [xopt,fopt,exitflag,output]=fminunc(.....)
  //
  //
  //   Parameters
  //   _f : a function, represents objective function of the problem 
  //   x1 : a scalar or vector(1 X 1), contains lower bound
  //   x2 : a scalar or vector(1 X 1), contains upper bound  
  //   options : a list, contains option for user to specify -Maximum iteration, Maximum CPU-time, TolX
  //			 Syntax for option- options= list("MaxIter", [---], "CpuTime", [---],TolX, [---]);
  //   		     Default Values for Options==> ("MaxIter", [1000000], "CpuTime", [1000000], TolX, [1e-4]);
  //   xopt : a vector of doubles, the computed solution of the optimization problem.
  //   fopt : a double, the function value at x.
  //   exitflag : Integer identifying the reason the algorithm terminated.
  //   output   : Structure containing information about the optimization.
  //
  //   Description
  //   Search the minimum of a single-variable function on open bounded interval specified by :
  //   find the minimum of f(x) such that 
  //
  //   <latex>
  //    \begin{eqnarray}
  //    &\mbox{min}_{x}
  //    & f(x)\\
  //    & \text{subject to} & x1 \< x \< x2 \\
  //    \end{eqnarray}
  //   </latex>
  //
  //   We are calling IPOpt for solving the unconstrained problem, IPOpt is a library written in C++. The code has been written by ​Andreas Wächter and ​Carl Laird.
  //
  // Examples
  //      //Find x in R^2 such that the rosenbrock function is minimum
  //      //f = 100*(x(2) - x(1)^2)^2 + (1-x(1))^2;
  //
  //      function y= _f(x)
  //   	   	y= 100*(x(2) - x(1)^2)^2 + (1-x(1))^2;
  //      endfunction
  //   	  x1=5;
  //	  x2=10;
  //      options=list("MaxIter", [1500], "CpuTime", [500], "TolX", [1e-6]);
  //      [xopt,fopt,exitflag,output]=fminunc(_f,x0,options)
  //
  //
  // Examples
  //      //Find x in R^2 such that the below function is minimum
  //      //f = x(1)^2 + x(2)^2
  //
  //      function y= _f(x)
  //   	   	y= x(1)^2 + x(2)^2;
  //      endfunction
  //   	  x1=-3;
  //	  x2=3;
  //      options=list("MaxIter", [100], "CpuTime", [10], "TolX", [1e-2]);
  //      [xopt,fopt]=fminunc(_f,x0,options)
  //
  // Authors
  // R.Vidyadhar , Vignesh Kannan
   	//To check the number of input and output argument
   	[lhs , rhs] = argn();
	
   	//To check the number of argument given by user
   	if ( rhs<3 | rhs>4 ) then
    		errmsg = msprintf(gettext("%s: Unexpected number of input arguments : %d provided while should be 3 or 4"), "fminbnd", rhs);
   		error(errmsg)
   	end
   
   	//Storing the 1st and 2nd Input Parameters  
   	_f = varargin(1);
   	x1 = varargin(2);
   	x2 = varargin(3);
   
   	s1=size(x1)
   	s2=size(x2)

   	//To check whether the 2nd and 3rd Input argument(x1,x2) is a column or row vector
   	if (s1(1)~=1 | s1(2)~=1 | s2(1)~=1 | s2(2)~=1)
	errmsg = msprintf(gettext("%s: Expected a double value(1x1 double matrix) for x1(Lower bound) and x2(Upper bound) "), "fminbnd");
    		error(errmsg)
   	end
   
   	//To check the difference between Upper and Lower Bound
   	if (x2-x1<=1e-6) then
   		errmsg = msprintf(gettext("%s: Difference between Upper Bound and Lower bound should be > 10^6) "), "fminbnd");
    		error(errmsg)
   	end

   	//To check, Whether Options is been entered by user  
   	if ( rhs<4 | size(varargin(4)) ==0 ) then
      		param = list(); 
   	else
      		param =varargin(4); //Storing the 3rd Input Parameter in intermediate list named 'param'
   	end
   
   	options = list("MaxIter",[3000],"CpuTime",[600],"TolX", [1e-4]);
      
   	//To check the User Entry for Options and storing it
   	for i = 1:(size(param))/2
       		select param(2*i-1)
    		case "MaxIter" then
          			options(2*i) = param(2*i);
       		case "CpuTime" then
          			options(2*i) = param(2*i);
        	case "TolX" then
          			options(2*i) = param(2*i); 
    		else
    	     	 	errmsg = msprintf(gettext("%s: Unrecognized parameter name ''%s''."), "fminbnd", param(2*i-1));
    	      		error(errmsg)
    		end
   	end

   	
   	//Defining a function to calculate Gradient or Hessian.
   	function y=_gradhess(x,t)
		if t==1 then
			y=numderivative(_f,x)
		else
			[grad,y]=numderivative(_f,x)
		end
   	endfunction
   
   	//Calling sci_solveminuncp by sending the inputted paramters 
   	[xopt,fopt,status,iter] = solveminbndp(_f,_gradhess,x1,x2,options);
   
   	//Calculating the values for output
   	xopt = xopt';
   	exitflag = status;
   	output = struct("Iterations", []);
   	output.Iterations = iter;

    	//To print Output Message
    	select status
    
    	case 0 then
        	printf("\nOptimal Solution Found.\n");
    	case 1 then
        	printf("\nMaximum Number of Iterations Exceeded. Output may not be optimal.\n");
    	case 2 then
        	printf("\nMaximum CPU Time exceeded. Output may not be optimal.\n");
    	case 3 then
        	printf("\nStop at Tiny Step\n");
    	case 4 then
        	printf("\nSolved To Acceptable Level\n");
    	case 5 then
        	printf("\nConverged to a point of local infeasibility.\n");
    	case 6 then
        	printf("\nStopping optimization at current point as requested by user.\n");
    	case 7 then
        	printf("\nFeasible point for square problem found.\n");
    	case 8 then 
        	printf("\nIterates diverging; problem might be unbounded.\n");
    	case 9 then
        	printf("\nRestoration Failed!\n");
    	case 10 then
        	printf("\nError in step computation (regularization becomes too large?)!\n");
    	case 12 then
        	printf("\nProblem has too few degrees of freedom.\n");
    	case 13 then
       		printf("\nInvalid option thrown back by IPOpt\n");
    	case 14 then
        	printf("\nNot enough memory.\n");
    	case 15 then
        	printf("\nINTERNAL ERROR: Unknown SolverReturn value - Notify IPOPT Authors.\n");
    	else
        	printf("\nInvalid status returned. Notify the Toolbox authors\n");
        	break;
    	end
    

endfunction
