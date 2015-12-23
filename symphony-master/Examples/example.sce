//fminunc
function y=fun(x)
y=-((x(1)-1)^2+(x(2)-1)^2);
endfunction



//FAILURE CASES
//Fails sometimes if starting point is a stationary point i.e. f'=0. (works for x1^2+x2^2 and (0,0) but fails for (x1-1)^2+(x2-1)^2 and (1,1))
//Fails when it converges to point of inflecion. So if function has a point of inflection nearby when compared to the local minimum the  function may fail to find the optimal value.

exec builder.sce
exec loader.sce

pt=[0];
options=list("MaxIter", [1500], "CpuTime", [500]);
[x,f,e,s]=fminbnd(fun,[-%inf,-%inf],[],options)

