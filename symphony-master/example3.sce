function y=_f(x)
	y=-x^1.5
endfunction

x1 = -10;
x2 = -5;
exec builder.sce
exec loader.sce

[xopt,fopt,exitflag,status] = fminbnd(_f,x1,x2)
