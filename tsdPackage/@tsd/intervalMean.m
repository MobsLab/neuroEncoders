function im = intervalMean(tsa, is, varargin)
% im = intervalCount(tsa, is, options) computes mean of tsa data in each of the intervals in is
%
% INPUTS:
% tsa: a tsd object
% is: an intervalSet
% OUTPUTS: 
% ic: a tsd object, where the timestamps correspond to each interval (see
% OPTIONS for possibilities) and the data gives the mean  of the data points in
% the tsd in each one of the intervals
% OPTIONS:
% 'Time': determines which time is selected for each interval, possible
% values are 
%     'start' (default): use start of intervals
%     'end': use end of intervals
%     'middle': use middle point of intervals
  
  
  
% copyright (c) 2004 Francesco P. Battaglia
% This software is released under the GNU GPL
% www.gnu.org/copyleft/gpl.html
  
  opt_varargin = varargin;
  
  
  time_opt_values = dictArray({ { 'start', []},
		                { 'end', []}, 
		                { 'middle', []} });
  defined_options = dictArray({ { 'Time', {'start', {'char'} } } } );
  
  getOpt;
  
  d = Data(tsa);
  
  if isempty(d)
    error('intervalMean not defined for tsd with empty data!');
  end
  
  
  n_dim = length(size(d));
  
% $$$   if ~isLinearArray(d)
    d = permute(d, [(2:n_dim) 1]); % transpose d so that time index will be
				   % major index, mkaing it easier to deal
				   % with it in C
% $$$   end
  
  
  im = intervalMean_c(Range(tsa), d, ...
		      Start(is, tsa.time_unit), End(is, tsa.time_unit));
  
% $$$   if ~isLinearArray(d)

    im = permute(im, [n_dim (1:(n_dim-1))]); % restore the right
                                             % dimension order  
% $$$   end
  
  
  switch Time
   case 'start'
    t_im = Start(is, tsa.time_unit);
   case 'end'
    t_im = End(is, tsa.time_unit);
   case 'middle'
    t_im = ( Start(is, tsa.time_unit) + End(is, tsa.time_unit) ) / 2;
  end
  
  im = tsd(t_im, im);