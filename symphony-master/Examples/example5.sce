function y=_f(x)
	y=0
	for i =1:6
		y=y+sin(x(i));
	end	

endfunction

	x1 = [-2, -2, -2, -2, -2, -2];
    x2 = [2, 2, 2, 2, 2, 2];

//exec builder.sce
//exec loader.sce
options=list("MaxIter",[1500],"CpuTime", [100],"TolX",[1e-6])

[xopt,fopt,exitflag,output,zl,zu] = fminbnd(_f,x1,x2,options)
	
