function [tso, ix] = RestrictThreshold(tsa, thr, varargin)
% [tso, ix] = Restrict_threshold(tsa, thr, varargin) Restricts tsd to
% values above (below) threshold
% 
% INPUTS: 
% tsa: a tsd object
% thr: a threshold value 
% OUTPUTS:
% tso: the restricted tsd
% ix (optional): the indices of the restricted tsd within tsa
% OPTIONS:
% 'Direction': possible values are 'Above' (default) 'Below'
  
% copyright (c) 2004 Francesco P. Battaglia
% This software is released under the GNU GPL
% www.gnu.org/copyleft/gpl.html

 opt_varargin = varargin;
 
 defined_options = dictArray({ { 'Direction', {'Above', { 'char' } } } ...
		   });
 
 getOpt;
 
 is = thresholdIntervals(tsa, thr, 'Direction', Direction);
 
 [tso, il] = Restrict(tsa, is);
 
 if nargout == 2
   ix= il;
 end
   
  
  