function y=_f(x)
	y=-(x(1)+x(2));
endfunction

x0 = [0.1,0.1];
A = [];
b = [];
Aeq = [];
beq = [];
no_nlic=2;
lb=[0,0];
ub=[];

//exec builder.sce
//exec loader.sce
function [y]=_nlc(x)
	y(1)=x(2)-x(1)^1.5;
	y(2)=-(x(1)+x(2));
endfunction

options=list("MaxIter", [1500], "CpuTime", [500], "GradObj", "OFF", "HessObj", "OFF","GradCon", "OFF");


function [y]= _conG(x)
	y(1)=2*(x(1)-1/3)
	y(2)=2*(x(1)-1/3)
	y(3)=-2*x(1)
	y(4)=x(2)
	y(5)=0
	y(6)=1
endfunction
[x,fval,exitflag,output,lambda,grad,hessian] = fmincon(_f,x0,A,b,Aeq,beq,lb,ub,no_nlic,_nlc)

