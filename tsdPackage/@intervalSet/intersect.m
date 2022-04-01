function O = intersect(varargin)
% O = intersect(O1, O2, ..., On) compute the intersection of intervalSet's
%
% INPUTS:
% O1, O2, ..., On: intervalSet objects
% OUTPUTS:
% O: intersection intervalSet object

% batta 2001 
% starting version
  
  
if nargin < 2
  error('Call with at least two arguments');
end

for i = 1:nargin
  if ~isa(varargin{i}, 'intervalSet')
    error('Arguments must be intervalSet');
  end
end


do_string = '[start, stop] = do_intersect(';
for i = 1:nargin
  do_string = [do_string 'Start(varargin{' num2str(i) ...
	       '}, ''ts''), End(varargin{' num2str(i) '}, ''ts''), ' ...
	      ];
end

do_string = [do_string(1:(end-2)) ' );'];

eval(do_string);

O = intervalSet(start, stop);

  
  
