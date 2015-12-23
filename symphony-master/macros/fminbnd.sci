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


function [xopt,fopt,exitflag,output,zl,zu] = fminbnd (varargin)
  // Find minimum of Multi-variable function on boundeed interval
  //
  //   Calling Sequence
  //   xopt = fminbnd(f,x1,x2)
  //   xopt = fminbnd(f,x1,x2,options)
  //   [xopt,fopt] = fminbnd(.....)
  //   [xopt,fopt,exitflag]= fminbnd(.....)
  //   [xopt,fopt,exitflag,output]=fminbnd(.....)
  //   [xopt,fopt,exitflag,output,zl,zu]=fminbnd(.....)
  //
  //
  //   Parameters
  //   f : a function, representing objective function of the problem 
  //   x1 : a vector, containing lower bound of the variables of size (1 X n) or (n X 1) where 'n' is the no. of Variables, where n is no. of Variables
  //   x2 : a vector, containing upper bound of the variables of size (1 X n) or (n X 1) or (0 X 0) where 'n' is the no. of Variables. If x2 is empty it means upper bound is +infinity
  //   options : a list, containing option for user to specify -Maximum iteration, Maximum CPU-time, TolX
  //			 Syntax for options- options= list("MaxIter", [---], "CpuTime", [---],TolX, [---]);
  //   		     Default Values for Options==> ("MaxIter", [3000], "CpuTime", [600], TolX, [1e-4]);
  //   xopt : a vector of doubles, containing the computed solution of the optimization problem.
  //   fopt : a scalar of double, containing the function value at x.
  //   exitflag : a scalar of integer, containing flag which denotes the reason for termination of algorithm
  //   output : a structure, containing information about the optimization.
  //   zl : a vector of doubles, containing lower bound multipliers
  //   zu : a vector of doubles, containing upper bound multipliers
  //
  //   Description
  //   Search the minimum of a multi-variable function on bounded interval specified by :
  //   find the minimum of f(x) such that 
  //
  //   <latex>
  //    \begin{eqnarray}
  //    &\mbox{min}_{x}
  //    & f(x)\\
  //    & \text{subject to} & x1 \ < x \ < x2 \\
  //    \end{eqnarray}
  //   </latex>
  //
  //   We are calling IPOpt for solving the unconstrained problem, IPOpt is a library written in C++.
  //
  // Examples
  //
  //	//Find x in R^6 such that it minimizes:
  //    //f(x)= sin(x1) + sin(x2) + sin(x3) + sin(x4) + sin(x5) + sin(x6)
  //	//-2 <= x1,x2,x3,x4,x5,x6 <= 2
  //
  //    //Objective function to be minimised
  //    function y=f(x)
  //		y=0
  //		for i =1:6
  //			y=y+sin(x(i));
  //		end	
  //	endfunction
  //
  //	//Variable bounds  
  //	x1 = [-2, -2, -2, -2, -2, -2];
  //    x2 = [2, 2, 2, 2, 2, 2];
  //
  //	//Options
  //	options=list("MaxIter",[1500],"CpuTime", [100],"TolX",[1e-6])
  //
  //    //Calling IPopt
  //	[x,fval,exitflag,output,zl,zu] =fminbnd(f, x1, x2, options)
  //
  // Examples
  //
  //	//Find x in R such that it minimizes:
  //    //f(x)= 1/x^2
  //	//0 <= x <= 1000
  //
  //    //Objective function to be minimised
  //    function y=f(x)
  //		y=1/x^2
  //	endfunction
  //
  //	//Variable bounds  
  //	x1 = [0];
  //    x2 = [1000];
  //
  //    //Calling IPopt
  //	[x,fval,exitflag,output,zl,zu] =fminbnd(f, x1, x2)
  //
  // Examples
  //
  //    The below Problem is an Unbounded problem:
  //	//Find x in R^2 such that it minimizes:
  //    //f(x)= -[(x1-1)^2 + (x2-1)^2]
  //	//-inf <= x1,x2 <= inf
  //
  //    //Objective function to be minimised
  //    function y=f(x)
  // 		y=-((x(1)-1)^2+(x(2)-1)^2);
  //	endfunction
  //
  //	//Variable bounds  
  //	x1 = [-%inf , -%inf];
  //    x2 = [];
  //
  //	//Options
  //	options=list("MaxIter",[1500],"CpuTime", [100],"TolX",[1e-6])
  //
  //    //Calling IPopt
  //	[x,fval,exitflag,output,zl,zu] =fminbnd(f, x1, x2, options)  
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
   	fun = varargin(1);
   	x1 = varargin(2);
   	x2 = varargin(3);

	function y = f(x)
		if(execstr('y=fun(x)','errcatch')==27) then
			errmsg = msprintf(gettext("%s: Please change the Objective function, there is division by zero Warning"), "fminbnd");
   			error(errmsg);
		end
		y=fun(x);
	endfunction
   	
   	//To check whether the 2nd Input argument (x1) is a Vector/Scalar
   	if (type(x1) ~= 1) then
   		errmsg = msprintf(gettext("%s: Expected Vector/Scalar for Lower Bound Vector (2nd Parameter)"), "fminbnd");
   		error(errmsg);
  	end
  	
  	if (size(x1,2)==0) then
        errmsg = msprintf(gettext("%s: Lower Bound (2nd Parameter) cannot be empty"), "fminbnd");
   		error(errmsg);
    end
    
   	if (size(x1,1)~=1) & (size(x1,2)~=1) then
      errmsg = msprintf(gettext("%s: Lower Bound (2nd Parameter) should be a vector"), "fminbnd");
      error(errmsg); 
   	elseif (size(x1,2)==1) then
   	 	x1=x1;
   	elseif (size(x1,1)==1) then
   		x1=x1';
   	end
   	s=size(x1)
   	
   	//To check whether the 3rd Input argument (x2) is a Vector/Scalar
   	if (type(x2) ~= 1) then
   		errmsg = msprintf(gettext("%s: Expected Vector/Scalar for Upper Bound Vector (3rd Parameter)"), "fminbnd");
   		error(errmsg);
  	end
   	
   	//To check for correct size and data of x2 (3rd paramter) and Converting it to Column Vector as required by Ipopt
    if (size(x2,2)==0) then
		x2 = repmat(%inf,s(1),1);
    end
    
    if (size(x2,1)~=1)& (size(x2,2)~=1) then
      errmsg = msprintf(gettext("%s: Upper Bound (3rd Parameter) should be a vector"), "fminbnd");
      error(errmsg); 
    elseif(size(x2,1)~=s(1) & size(x2,2)==1) then
   		errmsg = msprintf(gettext("%s: Upper Bound and Lower Bound are not matching"), "fminbnd");
   		error(errmsg);
   	elseif(size(x2,1)==s(1) & size(x2,2)==1) then
   	 	x2=x2;
   	elseif(size(x2,1)==1 & size(x2,2)~=s(1)) then
   		errmsg = msprintf(gettext("%s: Upper Bound and Lower Bound are not matching"), "fminbnd");
   		error(errmsg);
   	elseif(size(x2,1)==1 & size(x2,2)==s(1)) then
   		x2=x2';
   	end 
    
    //To check the contents of x1 & x2 (2nd & 3rd Parameter)
    
    for i = 1:s(2)
		if (x1(i) == %inf) then
		   	errmsg = msprintf(gettext("%s: Value of Lower Bound can not be infinity"), "fminbnd");
    		error(errmsg); 
  		end	
		if (x2(i) == -%inf) then
		   	errmsg = msprintf(gettext("%s: Value of Upper Bound can not be negative infinity"), "fminbnd");
    		error(errmsg); 
		end	
		if(x2(i)-x1(i)<=1e-6) then
			errmsg = msprintf(gettext("%s: Difference between Upper Bound and Lower bound should be atleast > 10^6 for variable No.= %d "), "fminbnd", i);
    		error(errmsg)
    	end
	end
	
	//To check the match between f (1st Parameter) & x0 (2nd Parameter)

   	if(execstr('init=f(x1)','errcatch')==21) then
		errmsg = msprintf(gettext("%s: Objective function and bounds did not match"), "fminbnd");
   		error(errmsg);
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
   	function y=gradhess(x,t)
		if t==1 then
			y=numderivative(f,x)
		else
			[grad,y]=numderivative(f,x)
		end
   	endfunction
   
   	//Calling sci_solveminuncp by sending the inputted paramters 
	
	[xopt,fopt,status,iter,cpu,obj_eval,dual,zl,zu] = solveminbndp(f,gradhess,x1,x2,options);
   
   	//Calculating the values for output
   	xopt = xopt';
   	exitflag = status;
   	output = struct("Iterations", [],"Cpu_Time",[],"Objective_Evaluation",[],"Dual_Infeasibility",[]);
   	output.Iterations = iter;
    output.Cpu_Time = cpu;
    output.Objective_Evaluation = obj_eval;
    output.Dual_Infeasibility = dual;

	//In the cases of the problem not being solved return NULL to the output matrices
	if( status~=0 & status~=1 & status~=2 & status~=4 & status~=7 ) then
		xopt=[]
		fopt=[]
		output = struct("Iterations", [],"Cpu_Time",[]);
		output.Iterations = iter;
    	output.Cpu_Time = cpu;
		zl=[];
		zu=[];
	end
	
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
