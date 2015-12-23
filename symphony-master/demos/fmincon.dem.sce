mode(1)
//
// Demo of fmincon.sci
//

//Find x in R^2 such that the below function is minimum
//f = x(1)^2 + 2*x(2)
//Starting Point: [0,0]
//Constraint 1, c1(x)==>x(1)^2+x(2)^2<=2
//Constraint 1, c2(x)==>x(1)^2+x(2)^2=1
//Constraint 1's Gradient c1'(x)=[2*x(1),2*x(2)]
//Constraint 2's Gradient c2'(x)=[2*x(1),2*x(2)]
halt()   // Press return to continue
 
function y= _f(x)
y= x(1)^2 + 2*x(2);
endfunction
halt()   // Press return to continue
 
x0=[0,0];
A=[];
b=[];
Aeq=[];
beq=[];
lb=[];
ub=[];
no_nlic=1;
halt()   // Press return to continue
 
function y= _nlc(x)
y(1)= x(1)^2 + x(2)^2 -2;
y(2)= x(1)^2 + x(2)^2 -1;
endfunction
halt()   // Press return to continue
 
options=list(("MaxIter", [100], "CpuTime", [60], "GradObj", "OFF", "HessObj", "OFF", "GradCon", "ON");
function [y]=_cg(x)
y(1)=2*x(1);
y(2)=2*x(2);
y(3)=2*x(1);
y(4)=2*x(2);
endfunction
halt()   // Press return to continue
 
[xopt,fopt]=fmincon(_f,x0,A,b,Aeq,beq,lb,ub,no_nlic,_nlc,options,_cg)
halt()   // Press return to continue
 
//========= E N D === O F === D E M O =========//
