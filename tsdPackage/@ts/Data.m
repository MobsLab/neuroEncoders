function t = Data(TS, key)
%   t = Data(TS, key) returns the timestamps of t
%  if key agrument is present, returns the Restrict-ed version of ts to
%  key, aligned 'prev'

  



% copyright (c) 2004 Francesco P. Battaglia
% This software is released under the GNU GPL
% www.gnu.org/copyleft/gpl.html


  if nargin == 1
    t = Range(TS.tsd);
  else
    t = Range(Restrict(TS.tsd, key, 'align', 'prev'));
  end
  
  