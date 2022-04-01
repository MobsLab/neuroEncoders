function t = unique(tsa)
% tout = unique(tin) eliminates points with the same timestamps
%   
% INPUTS:  
% tsa: a tsd object
  
% copyright (c) 2006 Francesco P. Battaglia
% This software is released under the GNU GPL
% www.gnu.org/copyleft/gpl.html


 tim = Range(tsa);
 [tt, ix, j] = unique(tim);


 t = subset(tsa, ix);
 