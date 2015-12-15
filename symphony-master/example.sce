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
//exec builder.sce
//exec loader.sce
options=list("MaxIter", [1500], "CpuTime", [500], "Gradient", "ON", "Hessian", "OFF");
//options=list("MaxIter", [1500], "CpuTime", [600])
[x,f,e,s,g,h]=fminunc(fun,pt,options,grad)
