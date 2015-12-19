function y=_f(x)
	y=100*(x(2) - x(1)*x(1))*(x(2) - x(1)*x(1)) + (1 - x(1))*(1 - x(1));
endfunction

x0 = [0.5,0];


//exec builder.sce
//exec loader.sce


options=list("MaxIter", [1500], "CpuTime", [500], "Gradient", "OFF", "Hessian", "OFF");
function y= _g(x)
	y= [-400*x(1)*x(2) + 400*x(1)^3 + 2*x(1)-2, 200*(x(2)-x(1)^2)];
endfunction

function y= _h(x)
	y= [1200*x(1)^2- 400*x(2) + 2, -400*x(1);-400*x(1), 200 ];
endfunction


[xopt,fopt,status,iter,gradient, hessian1] = fminunc(_f,x0,options)

