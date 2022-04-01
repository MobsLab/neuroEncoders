function ic = intervalRate(tsa, is, TimeUnits, varargin)
% ic = intervalRate(tsa, is, options) returns rate of tsa in each of the intervals in is
%
% INPUTS:
% tsa: a tsd object
% is: an intervalSet
% TimeUnits: the units for the rate, takes a time units, to be intended as
% the units of the returned rate to the minus one (until we get a real
% units system :) 
% OUTPUTS: 
% ic: a tsd object, where the timestamps correspond to each interval (see
% OPTIONS for possibilities) and the data gives the number of points in
% the tsd in each one of the intervals
% OPTIONS:
% 'time': determines which time is selected for each interval, possible
% values are 
%     'start' (default): use start of intervals
%     'end': use end of intervals
%     'middle': use middle point of intervals
  
  
% copyright (c) 2004 Francesco P. Battaglia
% This software is released under the GNU GPL
% www.gnu.org/copyleft/gpl.html
  
  
  
  if nargin == 2
    TimeUnits = time_units('s');
  else 
    TimeUnits = time_units(TimeUnits);
  end
    
  opt_varargin = varargin;
  
  time_opt_values = dictArray({ { 'start', []},
		                { 'end', []}, 
		                { 'middle', []} });
  defined_options = dictArray({ { 'time', {'start', {'char'} } } } );
  
  getOpt;

  
  t_ic = intervalCount(tsa, is, 'time', time);


  l = (length(is, TimeUnits));
  ic = tsd(t_ic.t, t_ic.data ./ l.data);
  
  
  