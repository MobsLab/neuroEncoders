function SO = merge(S)
% SO = merge(S) merges tsd's in tsdArray
% 
% if timestamps coincide in all the tsd's, it returns a single tsd in
% which all the data are cat-ed horizontally
  
  
% copyright (c) 2004 Francesco P. Battaglia
% This software is released under the GNU GPL
% www.gnu.org/copyleft/gpl.html


  
  for i = 2:length(S.C)
    if ~compatible_ts(S.C{1}, S.C{i}, 1e-10)
      error(['incompatible timestamps within tsdArray: ' num2str(i)]);
    end
  end
  
  ndim = length(size(Data(S.C{1})));
  
  d = Data(S.C{1});
  
  for i = 2:length(S.C)
    d = cat(ndim, d, Data(S.C{i}));
  end
  
  
  SO = tsd(Range(S.C{1}), d);
  