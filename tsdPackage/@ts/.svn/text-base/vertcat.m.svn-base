function s = vertcat(a, b, varargin);
% s = vertcat(a, b) overload of the [a ; b] operator
%
% arguments must be all tsd, the cat function is used 
  
% copyright (c) 2004 Francesco P. Battaglia
% This software is released under the GNU GPL
% www.gnu.org/copyleft/gpl.html

  if length(varargin) > 0
    for i = 1:length(varargin)
      if ~isa(varargin{i}, 'tsd')
	error('all arguments must be tsd''s');
      end
    end
  end
  
  
  s = cat(a,b);
  
  if length(varargin) > 0
    for i = 1:length(varargin)
      s = cat(s, varargin{i});
    end
  end
  