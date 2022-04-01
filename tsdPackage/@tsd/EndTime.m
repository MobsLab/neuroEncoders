function T1 = EndTime(tsa, tsflag)
%
% T1 = tsd/EndTime(tsd, tsflag)
%	returns last timestamp covered by tsa
%
%      tsflag: if 'ts' returns time in timestamps (default),
%              if 'sec' returns time in sec
%              if 'ms' returns time in ms



% ADR 1998
% version L4.0
% status PROMOTED

T1 = tsa.t(end);

units_out = tsa.time_unit;



if nargin == 2
  if isa(tsflag, 'char')
    units_out = time_units(tsflag);
  elseif isa(tsflag, 'units')
    units_out = tsflag;
  else
    error(['tsflag must be units object or corresponding abbreviation' ...
	   ' string']);
  end
end

cnvrt = convert(tsa.time_unit, units_out);

if cnvrt ~= 1
  T1 = cnvrt * T1;
end
