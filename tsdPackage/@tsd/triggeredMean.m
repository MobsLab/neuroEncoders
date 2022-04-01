function M = triggeredMean(tsa, t, nbBins,sc,smo,varargin);

%  USAGE:
%  	
%  	M = triggeredDataMat(TSD,T,N)
%  
%  INPUT:
%  	TSD: a tsd
%  	T: time of triggers
%  	N: nb f bins before and after (i.e. a total of 2*N+1 bins)
%  
%  OUTPUT:
%  	M: a matrix length(T) x (2*N+1)
%  
%  Adrien Peyrache, 2008

  
d = Data(tsa);
rg = Range(tsa);
rgR = Range(Restrict(tsa,t));
ix = find(ismember(rg,rgR));
M = [];

lmax = length(d);
i=1;
%  keyboard
while i<=length(ix) & (ix(i)+nbBins<lmax)
	if (ix(i)-nbBins>1)
		m = d(ix(i)-nbBins:ix(i)+nbBins,:)';
%  		keyboard
		if size(M)
			M = M+m;
		else
			M = m;
		end
	end
	i = i+1;
end

M = M/length(ix);

figure,imagesc([-nbBins:nbBins]*0.5,[sc(1):1/lmax:sc(2)],Smooth(M,[smo/2,smo])), axis xy