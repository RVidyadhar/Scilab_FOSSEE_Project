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



function [xopt,fopt,exitflag,output,lambda,gradient,hessian] = fmincon (varargin)
  	// Solves a Unconstrainted Optimization Problem
  	//
  	//   Calling Sequence
  	//   xopt = fmincon(_f,x0)
 	//   xopt = fmincon(_f,x0,options)
  	//   xopt = fmincon(_f,x0,options,_g)
  	//   xopt = fmincon(_f,x0,options,_h)
  	//   xopt = fmincon(_f,x0,options,_g,_h)
  	//   [xopt,fopt] = fmincon(.....)
  	//   [xopt,fopt,exitflag]= fmincon(.....)
  	//   [xopt,fopt,exitflag,output]= fmincon(.....)
  	//   [xopt,fopt,exitflag,output,gradient]=fmincon(.....)
  	//   [xopt,fopt,exitflag,output,gradient,hessian]=fmincon(.....)
  	//
  	//
  	//   Input Parameters:-
  	//   _f 	: a function, represents objective function of the problem 
  	//   x0 	: a vector of doubles, contains starting of variables.
  	//   options	: a list, contains option for user to specify -Maximum iteration, Maximum CPU-time, Gradient- ON (or) OFF &  Hessian- ON (or) OFF
  	//   Default Values for Options==> ("MaxIter", [1000000], "CpuTime", [1000000], "Gradient", "OFF", "Hessian", "OFF");
  	//   _g 	: a function, represents gradient function of the problem (Vector Form) 
  	//   _h 	: a function, represents hessian function of the problem  (Symmetric Matrix form)
  	// 
  	//   Output Parameters:-
  	//   xopt     : a vector of doubles, the computed solution of the optimization problem.
  	//   fopt     : a double, the function value at x.
  	//   exitflag : Integer identifying the reason the algorithm terminated.
  	//   output   : Structure containing information about the optimization.
  	//   gradient : a vector of doubles, contains the gradient of the optimized point.
  	//   hessian  : a matrix of doubles, contains the hessian of the optimized point.
  	//
  	//
  	//   We are calling IPOpt for solving the unconstrained problem, IPOpt is a library written in C++. The code has been written by ​Andreas Wächter and ​Carl Laird.
  	//   It searches the minimum of a unconstrained optimization problem. Find the below examples for reference:
  	//
  	//   Example-1:
  	//   //Find x in R^2 such that the rosenbrock function is minimum
  	//   f = 100*(x(2) - x(1)^2)^2 + (1-x(1))^2;
  	//
  	//   Defining Objective Function:
  	//      function y= _f(x)
  	//   	   	y= 100*(x(2) - x(1)^2)^2 + (1-x(1))^2;
  	//      endfunction
  	//   Defining Gradient Function:	
  	//      function y= _g(x)
 	//   	   	y= [-400*(x(2)-x(1)^2)*x(1)-2*(1-x(1)), 200*(x(2)-x(1)^2)]; //Row Vector is expected for gradient function
  	//      endfunction
  	//   Defining Hessian Function:
  	//       function y= _h(x)
  	//   	   	y= [1200*x(1)^2, -400*x(1);-400*x(1), 200 ]; //symmentric Matrix is expected for hessian function
  	//       endfunction
  	//   Setting Initial point:
  	//   	    x0=[2,7];
  	//   Setting Options--.(Syntax for option- options= list("MaxIter", [---], "CpuTime", [---], "Gradient", "ON/OFF", "Hessian", "ON/OFF");)
  	//        options=list("MaxIter", [1500], "CpuTime", [500], "Gradient", "ON", "Hessian", "ON");
  	//  
 	//   calling fmincon: 
  	//        [xopt,fopt,exitflag,output,gradient,hessian]=fmincon(_f,x0,options,_g,_h)
  	//
  	//
  	//   Example-2:
  	//   Find x in R^2 such that the below function is minimum
  	//   f = x(1)^2 + x(2)^2
  	//
  	//   Defining Objective Function:
  	//      function y= _f(x)
  	//   	   	y= x(1)^2 + x(2)^2;
  	//      endfunction
  	//   Defining Gradient Function:	
  	//      function y= _g(x)
  	//   	   	y= [2*x(1), 2*x(2)]; //Row Vector is expected for gradient function
  	//      endfunction
  	//   Setting Initial point:
  	//   	    x0=[2,7];
  	//   Setting Options--.(Syntax for option- options= list("MaxIter", [---], "CpuTime", [---], "Gradient", "ON/OFF", "Hessian", "ON/OFF");)
  	//        options=list("Gradient", "ON");
  	//  
  	//   calling fmincon: 
  	//        [xopt,fopt]=fmincon(_f,x0,options,_g)
 

	//To check the number of input and output argument
   	[lhs , rhs] = argn();
	
	//To check the number of argument given by user
   	if ( rhs<4 | rhs>14 ) then
    		errmsg = msprintf(gettext("%s: Unexpected number of input arguments : %d provided while it should be 4,6,8,10,11,12,13,14,15"), "fmincon", rhs);
    		error(errmsg)
   	end
    if (rhs==5 | rhs==7 | rhs==9) then
    	errmsg = msprintf(gettext("%s: 2Unexpected number of input arguments : %d provided while it should be 4,6,8,10,11,12,13,14,15"), "fmincon", rhs);
    	error(errmsg)
   	end
 
	//Storing the 1st and 2nd Input Parameters  
   	_f    	 = varargin(1);
   	x0   	 = varargin(2);
   	A    	 = varargin(3);
   	b    	 = varargin(4);
   	Aeq  	 = [];
   	beq  	 = [];
   	lb       = [];
   	ub       = [];
   	no_nlic  =[];
   	_nlc     = [];
   	//size(A);
   	if (rhs>4) then
   		Aeq  	 = varargin(5);
   		beq  	 = varargin(6);
   	end
   	if (rhs>6) then
   		lb       = varargin(7);
   		ub       = varargin(8);
   	end
   	if (rhs>8) then
   		no_nlic   = varargin(9);
   		_nlc      = varargin(10);
	end
	 
	//To check whether the 1st Input argument(_f) is a function or not
   	if (type(_f) ~= 13 & type(_f) ~= 11) then
   		errmsg = msprintf(gettext("%s: Expected function for Objective (1st Parameter) "), "fmincon");
   		error(errmsg);
   	end
   
	//To check whether the 2nd Input argument(x0) is a Vector/Scalar
   	if (type(x0) ~= 1) then
   		errmsg = msprintf(gettext("%s: Expected Vector/Scalar for Starting Point (2nd Parameter)"), "fmincon");
   		error(errmsg);
  	end
  	
  	//Returns "Invalid Index" Error if size of x0 is not matched with _f
   	if(execstr('init=_f(x0)','errcatch')==21) then
		errmsg = msprintf(gettext("%s: Objective function and x0 didnot match"), "fmincon");
   		error(errmsg);
	end
   	
  	//To check and convert the 2nd Input argument(x0) to row Vector 
   	if((size(x0,1)~=1) & (size(x0,2)~=1)) then
   		errmsg = msprintf(gettext("%s: Expected Row Vector or Column Vector for x0 (Initial Value) "), "fmincon");
   		error(errmsg);
    end
   	if(size(x0,2)==1) then
   		x0=x0';		//Converting x0 to row vector, if it is column vector
   	else 
   	 	x0=x0;		//Retaining the same, if it is already row vector
   	end   	 	
    s=size(x0);	
    
  	//To check whether the 3rd Input argument(A) is a Matrix/Vector
   	if (type(A) ~= 1) then
   		errmsg = msprintf(gettext("%s: Expected Matrix/Vector for Constraint Matrix A (3rd parameter)"), "fmincon");
   		error(errmsg);
  	end

	//To check for correct size of A(3rd paramter)
   	if(size(A,2)~=s(2) & size(A,2)~=0) then
   		errmsg = msprintf(gettext("%s: Expected Matrix of size (No of Linear Inequality Constraints X No of Variables) or an Empty Matrix for Linear Inequality Constraint coefficient Matrix A"), "fmincon");
   		error(errmsg);
   	end
   	s1=size(A);
   	
	//To check whether the 4th Input argument(b) is a Vector/Scalar
   	if (type(b) ~= 1) then
   		errmsg = msprintf(gettext("%s: Expected Vector/Scalar for b (4th Parameter)"), "fmincon");
   		error(errmsg);
  	end
  	
  	//To check for correct size of b(4th paramter)
    if(s1(2)==0) then
    	if(size(b,2)~=0) then
    		errmsg = msprintf(gettext("%s: As Linear Inequality Constraint coefficient Matrix A(3rd parameter) is empty, b(4th Parameter) should also be empty"), "fmincon");
   			error(errmsg);
   		end
   	else
   		if((size(b,1)~=1) & (size(b,2)~=1)) then
   			errmsg = msprintf(gettext("%s: Expected Non empty Vector for b (4th Parameter) for your Inputs "), "fmincon");
   			error(errmsg);
   		elseif(size(b,1)~=s1(1) & size(b,2)==1) then
   			errmsg = msprintf(gettext("%s: Expected Column Vector (No. of Linear inequality Constraints X 1) for b (4th Parameter) "), "fmincon");
   			error(errmsg);
   		elseif(size(b,1)==s1(1) & size(b,2)==1) then 
   	 		b=b;
   		elseif(size(b,1)==1 & size(b,2)~=s1(1)) then
   			errmsg = msprintf(gettext("%s: Expected Row Vector (1 X No. of Linear inequality Constraints) for b (4th Parameter) "), "fmincon");
   			error(errmsg);
   		elseif(size(b,1)==1 & size(b,2)==s1(1)) then
   			b=b';
   		end 
   	end
  	
  	//To check whether the 5th Input argument(A) is a Matrix/Vector
   	if (type(Aeq) ~= 1) then
   		errmsg = msprintf(gettext("%s: Expected Matrix/Vector for Equality Constraint Matrix Aeq (5th Parameter)"), "fmincon");
   		error(errmsg);
  	end
  	
  	//To check for correct size of Aeq(5th paramter)
   	if(size(Aeq,2)~=s(2) & size(Aeq,2)~=0) then
   		errmsg = msprintf(gettext("%s: Expected Matrix of size (m X n) or (0 X 0) for Linear equality Constraint coefficient Matrix Aeq where m is no of Linear equality Constraints & n is no.of Variables"), "fmincon", rhs);
   		error(errmsg);
   	end
   	s2=size(Aeq);

	//To check whether the 6th Input argument(b) is a Vector/Scalar
   	if (type(beq) ~= 1) then
   		errmsg = msprintf(gettext("%s: Expected Vector/Scalar for beq (6th Parameter)"), "fmincon");
   		error(errmsg);
  	end
  	
  	//To check for correct size of beq(6th paramter)
    if(s2(2)==0) then
    	if(size(beq,2)~=0) then
    		errmsg = msprintf(gettext("%s: As Linear Equality Constraint coefficient Matrix Aeq(5th parameter) is empty, beq(6th Parameter) should also be empty"), "fmincon");
   			error(errmsg);
   		end
   	else
   		if((size(beq,1)~=1) & (size(beq,2)~=1)) then
   			errmsg = msprintf(gettext("%s: Expected Non empty Vector for beq (6th Parameter) for your Inputs"), "fmincon");
   			error(errmsg);
   		elseif(size(beq,1)~=s2(1) & size(beq,2)==1) then
   			errmsg = msprintf(gettext("%s: Expected Column Vector (No. of Linear Equality Constraints X 1) for beq (6th Parameter) "), "fmincon");
   			error(errmsg);
   		elseif(size(beq,1)==s2(1) & size(beq,2)==1) then 
   	 		beq=beq;
   		elseif(size(beq,1)==1 & size(beq,2)~=s1(1)) then
   			errmsg = msprintf(gettext("%s: Expected Row Vector (1 X No. of Linear Equality Constraints) for beq (6th Parameter) "), "fmincon");
   			error(errmsg);
   		elseif(size(beq,1)==1 & size(beq,2)==s1(1)) then
   			beq=beq';
   		end 
   	end
   	
  	
  	//To check whether the 7th Input argument(lb) is a Vector/Scalar
   	if (type(lb) ~= 1) then
   		errmsg = msprintf(gettext("%s: Expected Vector/Scalar for Lower Bound Vector (7th Parameter)"), "fmincon");
   		error(errmsg);
  	end
  	
  	//To check whether the 8th Input argument(ub) is a Vector/Scalar
   	if (type(ub) ~= 1) then
   		errmsg = msprintf(gettext("%s: Expected Vector/Scalar for Upper Bound Vector (8th Parameter)"), "fmincon");
   		error(errmsg);
  	end
  	
  	//To check for correct size and data of lb(7th paramter)
   	if (size(lb,2)==0) then
        lb = repmat(-%inf,1,s(2));
    end
    
   	if (size(lb,1)~=1) & (size(lb,2)~=1) then
      errmsg = msprintf(gettext("%s: Lower Bound(7th Parameter) should be a vector"), "fmincon");
      error(errmsg); 
    elseif(size(lb,1)~=s(2) & size(lb,2)==1) then
   		errmsg = msprintf(gettext("%s: Expected Column Vector (No. of Variables X 1) for lower bound (7th Parameter) "), "fmincon", rhs);
   		error(errmsg);
   	elseif(size(lb,1)==s(2) & size(lb,2)==1) then
   	 	lb=lb';
   	elseif(size(lb,1)==1 & size(lb,2)~=s(2)) then
   		errmsg = msprintf(gettext("%s: Expected Row Vector (1 X No. of Variables) for lower bound (7th Parameter) "), "fmincon", rhs);
   		error(errmsg);
   	elseif(size(lb,1)==1 & size(lb,2)==s(2)) then
   		lb=lb;
   	end 
   	
   	//To check for correct size and data of lb(7th paramter)
    if (size(ub,2)==0) then
        ub = repmat(%inf,1,s(2));
    end
    
    if (size(ub,1)~=1)& (size(ub,2)~=1) then
      errmsg = msprintf(gettext("%s: Upper Bound(8th Parameter) should be a vector"), "fmincon");
      error(errmsg); 
    elseif(size(ub,1)~=s(2) & size(ub,2)==1) then
   		errmsg = msprintf(gettext("%s: Expected Column Vector (No. of Variables X 1) for upper bound (8th Parameter) "), "fmincon", rhs);
   		error(errmsg);
   	elseif(size(ub,1)==s(2) & size(ub,2)==1) then
   	 	ub=ub';
   	elseif(size(ub,1)==1 & size(ub,2)~=s(2)) then
   		errmsg = msprintf(gettext("%s: Expected Row Vector (1 X No. of Variables) for beq (8th Parameter) "), "fmincon", rhs);
   		error(errmsg);
   	elseif(size(ub,1)==1 & size(ub,2)==s(2)) then
   		ub=ub;
   	end 
    
    //To check the contents of lb & ub (7th & 8th Parameter)
    for i = 1:s(2)
		if (lb(i) == %inf) then
		   	errmsg = msprintf(gettext("%s: Value of Lower Bound can not be infinity"), "fmincon");
    		error(errmsg); 
  		end	

		if (ub(i) == -%inf) then
		   	errmsg = msprintf(gettext("%s: Value of Upper Bound can not be negative infinity"), "fmincon");
    		error(errmsg); 
		end	
		if(ub(i)-lb(i)<=1e-6) then
			errmsg = msprintf(gettext("%s: Difference between Upper Bound and Lower bound should be atleast > 10^6 for variable No.= %d "), "fminbnd", i);
    		error(errmsg)
    	end
	end
  	
  	//To check whether the 9th Input argument(no_nlic) is a Scalar
   	if (type(no_nlic) ~= 1) then
   		errmsg = msprintf(gettext("%s: Expected Scalar for no. of non Linear inequality constraints (9th Parameter)"), "fmincon");
   		error(errmsg);
  	end
  	
  	//To check whether the 10th Input argument(_nlc) is a function or an empty Matrix
   	if (size(no_nlic,2)~=0) then
   		if (type(_nlc) ~= 13 & type(_nlc) ~= 11) then
   			errmsg = msprintf(gettext("%s: Expected function for non Linear Constraints (10th Parameter) "), "fmincon");
   			error(errmsg);
   		end
	else
		if (type(_nlc)~=1) then
			errmsg = msprintf(gettext("%s: As no. of Non Linear Inequality Constraints(9th Parameter) is empty, non linear constraint function(10th Parameter) should also be empty"), "fmincon");
   			error(errmsg);
   		end
   		if (size(_nlc,2)~=0) then
   		errmsg = msprintf(gettext("%s: As no. of Non Linear Inequality Constraints(9th Parameter) is empty, non linear constraint function(10th Parameter) should also be empty"), "fmincon");
   			error(errmsg);
   		end
   	end
    
    //To check for correct size of no_nlic(9th paramter)
	no_nlc=1
	if (size(no_nlic,2)==0)
		no_nlc=0;
		no_nlic=0;
	elseif (size(no_nlic,1)~=1 | size(no_nlic,2)~=1) then
		errmsg = msprintf(gettext("%s: Expected scalar or Empty matrix for no. of non linear inequality constraints(9th Parameter) in the Non linear constraint function"), "fmincon");
    	error(errmsg); 
	end
	
	//Returns "Invalid Index" Error if size of x0 is not matched with _nlc(10th Parameter)
   	if (no_nlc==1) then
   		if(execstr('init1=_nlc(x0)','errcatch')==21)
			errmsg = msprintf(gettext("%s: Non-Linear Constraint function(9th Parameter) and x0(2nd Parameter) didnot match "), "fmincon");
   			error(errmsg);
		end
   		no_nlc=size(init1,1);
   	end
   	
   	//To check, Whether Options is been entered by user   
   	if ( rhs<11  ) then
      		param = list();
       else
      		param =varargin(11); //Storing the 3rd Input Parameter in intermediate list named 'param'
    end
   
	//If Options is entered then checking its type for 'list'   
   	if (type(param) ~= 15) then
   		errmsg = msprintf(gettext("%s: 3rd Input parameter should be a list (ie. Options) "), "fmincon");
   		error(errmsg);
   	end
   
	//If Options is entered then checking whether even no. of entires are entered   
   	if (modulo(size(param),2)) then
		errmsg = msprintf(gettext("%s: Size of parameters should be even"), "fmincon");
		error(errmsg);
   	end

	//To set Default Value for Options, If User Doesn't enter Options
   	options = list(..
      		"MaxIter"     , [1000000], ...
      		"CpuTime"   , [1000000] ...
      		);

	//Flags to check whether Gradient is "ON"/"OFF" & Hessian is "ON"/"OFF" 
   	flag1=0;
   	flag2=0;
   	flag3=0;
   	_fg=[];
   	_fh=[];
   	_cg=[];
   	
 
	//To check the User Entry for Options and storing it
   	for i = 1:(size(param))/2
       		select param(2*i-1)
    		case "MaxIter" then
          			options(2*i) = param(2*i);    //Setting the Maximum Iteration as per user entry
       		case "CpuTime" then
          			options(2*i) = param(2*i);    //Setting the Maximum CPU Time as per user entry
        	case "GradObj" then
        			if (param(2*i)=="ON") then
        				//To check whether the user has provided Gradient function if Gradient Option is "ON"
        				if (rhs<12) then      
				     		errmsg = msprintf(gettext("%s: Gradient function of Objective is missing, but GradObj=ON"), "fmincon");
				    		error(errmsg);     			
        				else
        				//This flag1 is activated(ie. =1) if Gradient is supplied
        					pos_fg=12;
        					flag1=1;
        					_fg=varargin(12);
        				end
        			//To check whether Wrong entry(other than ON/OFF) is entered
        			elseif (param(2*i)~="ON" & param(2*i)~="OFF") then    
        				errmsg = msprintf(gettext("%s: Options for GradObj should be either ON or OFF"), "fmincon");
					error(errmsg);     	
        			end
        	case "HessObj" then
        			if (param(2*i)=="ON") then
        				//To check whether the user has provided Hessian function if Hessian Option is "ON"
						if (flag1==0) then
							if (rhs<12) then    
				     			errmsg = msprintf(gettext("%s: Hessian function of Objective is missing, but HessObj=ON"), "fmincon");
				     			error(errmsg);		
        					else
        					//This flag is activated(ie. =1) if Hessian is supplied
        						pos_fh=12;
        						flag2=1;
        						_fh=varargin(12);
        			    	end         			
        				elseif (flag1==1) then
							if (rhs<13) then    
				     			errmsg = msprintf(gettext("%s: Hessian function of Objective is missing, but HessObj=ON"), "fmincon");
				     			error(errmsg);     			
        					else
        					//This flag is activated(ie. =1) if Hessian is supplied
        						pos_fh=13;
        						flag2=1;
        						_fh=varargin(13);
        			    	end
        			    end       
        			//To check whether Wrong entry(other than ON/OFF) is entered	            
        			elseif (param(2*i)~="ON" & param(2*i)~="OFF") then    
        				errmsg = msprintf(gettext("%s: Options for HessObj should be either ON or OFF"), "fmincon");
					error(errmsg);   
        			end
        	case "GradCon" then
        			if (param(2*i)=="ON") then
        				//To check whether the user has provided Gradient function if Gradient Option is "ON"
        				if (flag1==0 & flag2==0) then
        					if (rhs<12) then      
				     			errmsg = msprintf(gettext("%s: Gradient function of Non-Linear Constraint is missing, but GradCon=ON"), "fmincon");
				    			error(errmsg);     			
        					else
        						pos_cg=12;
        						flag3=1;
        						_cg=varargin(12);
        					end
        				elseif((flag1==1 & flag2==0) |(flag1==0 & flag2==1) ) then
        					if (rhs<13) then      
				     			errmsg = msprintf(gettext("%s: Gradient function of Constraints is missing, but GradCon=ON"), "fmincon");
				    			error(errmsg);    			
        					else
        						pos_cg=13;
        						flag3=1;
        						_cg=varargin(13);
        					end
        				elseif(flag1==1 & flag2==1) then
        					if (rhs<14) then      
				     			errmsg = msprintf(gettext("%s: Gradient function of Constraints is missing, but GradCon=ON"), "fmincon");
				    			error(errmsg);      			
        					else
        						pos_cg=14;
        						flag3=1;
        						_cg=varargin(14);
        					end        					
        				end        				      
        				//To check whether Wrong entry(other than ON/OFF) is entered
        			elseif (param(2*i)~="ON" & param(2*i)~="OFF") then    
        				errmsg = msprintf(gettext("%s: Options for GradCon should be either ON or OFF, but GradCon=ON"), "fmincon");
						error(errmsg);     	
        			end
        	else
    	      		errmsg = msprintf(gettext("%s: Unrecognized parameter name ''%s''."), "fmincon", param(2*i-1));
    	      		error(errmsg)
    		end
   	end
   
   
	//Defining a function to calculate Gradient or Hessian if the respective user entry is OFF 
   	function y=_gradhess(x,t)
		if t==1 then	//To return Gradient
			y=numderivative(_f,x)		
		elseif t==2 then		//To return Hessian]n
			[grad,y]=numderivative(_f,x)
		elseif t==3 then	//To return Gradient
			y=numderivative(_nlc,x)		
		elseif t==4 then		//To return Hessian]n
			[grad,y]=numderivative(_nlc,x)
		end
   	endfunction 
   
   if (flag1==0 & flag2==0 & flag3==0)
   		if(rhs>11) then
        	errmsg = msprintf(gettext("%s: Only 11 Inputs are Needed for this option(GradObj=OFF, HessObj=OFF, GradCon=OFF), but %d were recorded"), "fmincon",rhs);
			error(errmsg); 
		end
   elseif ((flag1==1 & flag2==0 & flag3==0) | (flag1==0 & flag2==1 & flag3==0) | (flag1==0 & flag2==0 & flag3==1)) then
  		if(rhs>12) then
        	errmsg = msprintf(gettext("%s: Only 12 Inputs were needed for this option, but %d were recorded"), "fmincon",rhs);
			error(errmsg);
		end
   elseif ((flag1==1 & flag2==1 & flag3==0) | (flag1==0 & flag2==1 & flag3==1) | (flag1==1 & flag2==0 & flag3==1)) then
   		if(rhs>13) then
        	errmsg = msprintf(gettext("%s: Only 13 Inputs were needed for this option, but %d were recorded"), "fmincon",rhs);
			error(errmsg);
		end
   end
	     	
   if (flag1==1) then
   		if (type(_fg) ~= 13 & type(_fg) ~= 11) then
  			errmsg = msprintf(gettext("%s: Expected function for Gradient of Objective, since GradObj=ON"), "fmincon");
   			error(errmsg);
   		end
   		if(execstr('sample_fg=_fg(x0)','errcatch')==21)
			errmsg = msprintf(gettext("%s: Gradient function of Objective and x0 didnot match "), "fmincon", rhs);
   			error(errmsg);
		end
		if (size(sample_fg,1)~=1| size(sample_fg,2)~=s(2)) then
   			errmsg = msprintf(gettext("%s: Wrong Input for Objective Gradient function(%dth Parameter)---->Row Vector function is Expected"), "fmincon",pos_fg);
   			error(errmsg);
   		end
   	end
   	if (flag2==1) then
   		if (type(_fh) ~= 13 & type(_fh) ~= 11) then
  			errmsg = msprintf(gettext("%s: Expected function for Hessian of Objective, since HessObj=ON"), "fmincon");
   			error(errmsg);
   		end
   		if(execstr('sample_fh=_fh(x0)','errcatch')==21)
			errmsg = msprintf(gettext("%s: Hessian function of Objective and x0 didnot match "), "fmincon", rhs);
   			error(errmsg);
		end
   		if(size(sample_fh,1)~=s(2) | size(sample_fh,2)~=s(2)) then
   			errmsg = msprintf(gettext("%s: Wrong Input for Objective Hessian function(%dth Parameter)---->Symmetric Matrix function is Expected "), "fmincon",pos_fh);
   			error(errmsg);
   		end
   	end
   	if (flag3==1) then
   		if (type(_cg) ~= 13 & type(_cg) ~= 11) then
  			errmsg = msprintf(gettext("%s: Expected function for Gradient of Constraint function,since GradCon=ON"), "fmincon");
   			error(errmsg);
   		end
   		if(execstr('sample_cg=_cg(x0)','errcatch')==21)
			errmsg = msprintf(gettext("%s: Gradient function of Constraint and x0 didnot match "), "fmincon", rhs);
   			error(errmsg);
		end
		sample_cg=_cg(x0)
   		if(size(sample_cg,1)~=no_nlc*s(2) | size(sample_cg,2)~=1) then
   			errmsg = msprintf(gettext("%s:  Wrong Input for Constraint Gradient function(%dth Parameter)---->Vector function is Expected (Refer Help)"), "fmincon",pos_cg);
   			error(errmsg);
   		end
   	end
   	disp(_f,_gradhess,x0,options,A,b,Aeq,beq,lb,ub,no_nlc,no_nlic,_nlc,flag1,_fg,flag2,_fh,flag3,_cg);
   	
  [xopt,fopt,status,iter,lambda,gradient,hessian1] = solveminconp (_f,_gradhess,A,b,Aeq,beq,lb,ub,no_nlc,no_nlic,_nlc,flag1,_fg,flag2,_fh,flag3,_cg,x0,options)		
   
	//Calculating the values for output

   	
   	//disp(_f,_gradhess,x0,options,A,b,Aeq,beq,lb,ub,no_nlc,no_nlic,_nlc,flag1,_fg,flag2,_fh,flag3,_cg);
   	
   	
   	xopt = xopt';
   exitflag = status;
  output = struct("Iterations", []);
   output.Iterations = iter;

    	//Converting hessian of order (1 x (numberOfVariables)^2) received from Ipopt to order (numberOfVariables x numberOfVariables)
    s=size(gradient)
    	for i =1:s(2)
    	for j =1:s(2)
			hessian(i,j)= hessian1(j+((i-1)*s(2)))
		end
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
    	//if (flag==2 |flag==3 |flag==4) then
		//disp("||||||Please Make sure you have entered Correct Functions for Gradient or Hessian -->Scilab Will Calculate Based on your input only||||||");
    	//end	
endfunction
