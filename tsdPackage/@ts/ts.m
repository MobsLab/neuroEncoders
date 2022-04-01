function tsa = ts(t, varargin)
%
% ts = ts(t)
% an object that represent a time series of events
% is a subclass of tsd, with empty Data, reimplements only the Data
% function, for compatibility issues. 
  
% copyright (c) 2004 Francesco P. Battaglia
% This software is released under the GNU GPL
% www.gnu.org/copyleft/gpl.html

  
  
  
  
if nargin == 0
   ts_tsd = tsd;
   
else
  ts_tsd = tsd(t, [], varargin{:}); % note that in case of constructor called with no
                       % arguments returns a tsd with NaN timestamps,
                       % instead of empty timestamps as in ADR's case
end

tsa.type = 'ts';
  
tsa = class(tsa, 'ts', ts_tsd);

