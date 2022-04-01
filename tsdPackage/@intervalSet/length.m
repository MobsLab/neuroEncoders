function L = length(O, TimeUnits, varargin)
% L = length(O, TimeUnits) length if each interval in the set
% INPUTS:
% O: an intervalSet object
% TimeUnits: a units object or the abbreviation string
% OUTPUT:
% L = a tsd object , where the timestamps correspond to each interval (see
% OPTIONS for possibilities) and the data gives the length of each interval
% OPTIONS:
% 'time': determines which time is selected for each interval, possible
% values are 
%     'start' (default): use start of intervals
%     'end': use end of intervals
%     'middle': use middle point of intervals
  
% copyright (c) 2004 Francesco P. Battaglia
% This software is released under the GNU GPL
% www.gnu.org/copyleft/gpl.html
  

  
  if nargin == 1
    TimeUnits = time_units('ts');
  else 
    TimeUnits = time_units(TimeUnits);
  end
  
  opt_varargin = varargin;
  
  
  time_opt_values = dictArray({ { 'start', []},
		                { 'end', []}, 
		                { 'middle', []} });
  defined_options = dictArray({ { 'time', {'start', {'char'} } } } );
  
  getOpt;
  
  l = End(O, TimeUnits) - Start(O, TimeUnits);
  
  switch time
   case 'start'
    t_ic = Start(O, TimeUnits);
   case 'end'
    t_ic = End(O, TimeUnits);
   case 'middle'
    t_ic = ( Start(O, TimeUnits) + End(O, TimeUnits) ) / 2;
  end
  
  L = tsd(t_ic, l);  
  
  