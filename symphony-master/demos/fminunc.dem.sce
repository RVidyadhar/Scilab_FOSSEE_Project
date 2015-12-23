mode(1)
//
// Demo of fminunc.sci
//

//Find x in R^2 such that it minimizes rosenbrock function
//f = 100*(x(2) - x(1)^2)^2 + (1-x(1))^2
halt()   // Press return to continue
 
function y= _f(x)
y= 100*(x(2) - x(1)^2)^2 + (1-x(1))^2;
endfunction
function y= _g(x)
y= [-400*(x(2)-x(1)^2)*x(1)-2*(1-x(1)), 200*(x(2)-x(1)^2)]; //Row Vector is expected for gradient function
endfunction
function y= _h(x)
y= [1200*x(1)^2, -400*x(1);-400*x(1), 200 ]; //symmentric Matrix is expected for hessian function
endfunction
x0=[2,7];
options=list("MaxIter", [1500], "CpuTime", [500], "Gradient", "ON", "Hessian", "ON");
[xopt,fopt,exitflag,output,gradient,hessian]=fminunc(_f,x0,options,_g,_h)
halt()   // Press return to continue
 
halt()   // Press return to continue
 
//Find x in R^2 such that the below function is minimum
//f = x(1)^2 + x(2)^2
halt()   // Press return to continue
 
function y= _f(x)
y= x(1)^2 + x(2)^2;
endfunction
x0=[2,1];
[xopt,fopt]=fminunc(_f,x0)
halt()   // Press return to continue
 
//========= E N D === O F === D E M O =========//
