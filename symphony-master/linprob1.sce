function y=_f(x)
	y=-x(1)-x(2)/3;
endfunction

x0=[0,0];
A = [1 1
    1 1/4
    1 -1
    -1/4 -1
    -1 -1
    -1 1];
b=[2 1 2 1 -1 2];
Aeq=[];
beq=[];
lb=[]
ub=[]
no_nlic=1;

options=list("MaxIter", [1500], "CpuTime", [500], "GradObj", "OFF", "HessObj", "OFF","GradCon", "OFF");

[x,fval,exitflag,output,lambda,grad,hessian] =fmincon(_f, x0,A,b)