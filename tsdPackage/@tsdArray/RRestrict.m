function S1 = RRestrict(S0,is)

%  tsaO = Restrict(tsa,is) Restrict function applied to each of the tsd of the tsdArray
%  Adrien Peyrache 2007

S1={};

for i=1:length(S0)
	
	Sr = S0.C{i};
	S1 = [S1;{Restrict(Sr,is)}];

end

S1 = tsdArray(S1);
	