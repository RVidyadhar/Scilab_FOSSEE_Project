//fminunc
function y=fun(x)
y=x(1)^3+x(2)^3;
endfunction

function y=grad(x)
y=[3*x(1)^2,3*x(2)^2];
endfunction

function y=hess(x)
y=[6*x(1),0;0,6*x(2)]
endfunction

pt=[1,2];
options=list("MaxIter", [1500], "CpuTime", [500], "Gradient", "ON", "Hessian", "ON");
[x,f,e,s,g,h]=fminunc(fun,pt,options,grad,hess)


//fminbnd
function y=fun1(x)
y=x^3-2*x-5
endfunction

options1=list("MaxIter",[100],"CpuTime", [100])
[x1,f1,e1,s1]=fminbnd(fun1,0,2,options1)

