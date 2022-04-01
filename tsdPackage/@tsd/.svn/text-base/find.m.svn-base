function O = find(tsa, find_string)
% O = find(tsa, find_string) selects points in tsd based on condition
%
% INPUTS:
% tsa: a tsd object
% find_string: a string that represent a condition, in which the data is
% substituted by the symbol 'Td', and the timestamps   
%  by the symbol 'Tt'
% OUTPUTS:
% O: the resulting tsd object
  
% copyright (c) 2004 Francesco P. Battaglia
% This software is released under the GNU GPL
% www.gnu.org/copyleft/gpl.html

  
  fs = substr(find_string, 'Td', 'tsa.data');
  fs = substr(fs, 'Tt', 'tsa.t');
  
  ix = eval(['find( ' fs ');']);
  
  O = tsd(tsa.t(ix), SelectAlongFirstDimension(tsa.data, ix));