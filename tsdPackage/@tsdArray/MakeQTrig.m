function Q = MakeQfromS(S, peaks, n)

% Q = MakeQTrig(S, peaks, n)
% 
% 
% INPUTS:
%    S - a cell array of ts objects
%    peaks - a ts of trigger time 
%    n - number of bins between each trigger time
%
% OUTPUTS:
%    Q - a ctsd in which the main structure is a |t| x nCells histogram of firing rates

% A Peyrache 2008

t = Range(peaks);
rg = [];

for i=1:length(t)-1

	ts = t(i);
	te = t(i+1);
	dt = (te-ts)/n;
	dt = [ts:dt:te-dt];
	rg = [rg;dt'];

end

S = Restrict(S,intervalSet(t(1),t(end)));
dQ = zeros(length(rg),length(S));

for i=1:length(S)
	spk = Data(S.C{i});
	l = length(spk);

	ix =1;
	while ix<l
		
		pos = binsearch_floor(rg,spk(ix));
		if spk(ix)<rg(pos+1)
			while spk(ix)<rg(pos+1) & ix<l 
				dQ(pos,i) = dQ(pos,i)+1;
				ix = ix+1;
			end
		else
			ix = l;
		end
	end

end

Q = tsd(rg,dQ);