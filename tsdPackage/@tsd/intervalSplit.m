function S = intervalSplit(tsa, is, varargin);
%  S = intervalSplit(tsa, is, optionName1, optionValue1, ...) returns a cell array of tsd object, one for each interval in is
% 
% INPUTS
% tsa: a tsd object
% is: an intervalSet object
% OUTPUTS:
% S: a cell array of tsd object one for each array 
% OPTIONS:
% 'OffsetStart', realigns timestamps so that beginning of interval
% corresponds to OptionValue
% 'OffsetEnd' realigns timestamps so that end of interval
% corresponds to OptionValue
% no realignment by default.
  
% copyright (c) 2004 Francesco P. Battaglia
% This software is released under the GNU GPL
% www.gnu.org/copyleft/gpl.html
  
  defined_options = dictArray({ 
      { 'OffsetStart', {NaN, {'numeric'} } }, 
      { 'OffsetEnd', {NaN, {'numeric'} } },
		   } );
  
  opt_varargin = varargin;
  
  getOpt;
  
  if isfinite(OffsetStart) & isfinite(OffsetEnd)
      error('Cannot specify both OffsetStart and OffsetEnd');
  end

  S = cell(0,1);

  if length(Start(is)) > 0
      [S_start, S_end] = intervalSplit_c(Range(tsa), Start(is, tsa.time_unit), ...
          End(is, tsa.time_unit));

      realign = zeros(size(Start(is)));

      if isfinite(OffsetStart)
          realign = OffsetStart - Start(is, tsa.time_unit);
      elseif isfinite(OffsetEnd)
          realign = OffsetEnd - End(is, tsa.time_unit);
      end



      for i = 1:length(S_start)
          if S_end(i)-S_start(i) > 0
              s = subset(tsa, (S_start(i)):(S_end(i)));
              s = tsd(s.t+realign(i), s.data);
          else
              s = tsd([], []);
          end

          S{i} = s;
      end
  end
  S = tsdArray(S);

    
