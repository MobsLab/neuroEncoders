function m = mean(tsa)
% r = rate(tsa) returns the mean  of data in tsd object
%
% INPUTS:
% tsa: a tsd object
% OUTPUTS:
% m: the rate   
  
% copyright (c) 2004 Francesco P. Battaglia
% This software is released under the GNU GPL
% www.gnu.org/copyleft/gpl.html

m = mean(tsa.data, 1);
