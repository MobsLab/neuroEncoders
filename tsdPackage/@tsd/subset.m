function tss = subset(tsa, ix);
% tss = subset(tsa, ix); returns tsd of a subset of input
%
% INPUTS:
% tsa: an input object
% ix: an array of indices 
% OUTPUTS: 
% tss: a tsd containing the point in tsa indicated by ix
 
  
% copyright (c) 2004 Francesco P. Battaglia
% This software is released under the GNU GPL
% www.gnu.org/copyleft/gpl.html
  
  
  t = Range(tsa);
  tss = tsd(t(ix), SelectAlongFirstDimension(Data(tsa), ix));
  
  