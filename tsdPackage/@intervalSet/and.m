function s = and(a, b)
% s = and(a, b) overload of the & operator
%
% defined as the intersection of the two intervalSet

% copyright (c) 2004 Francesco P. Battaglia
% This software is released under the GNU GPL
% www.gnu.org/copyleft/gpl.html

  s = intersect(a, b);
  
  