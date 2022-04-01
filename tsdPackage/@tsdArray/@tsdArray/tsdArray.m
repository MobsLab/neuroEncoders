function O = tsdArray(varargin)
% write docs  
  
  C = {};
  doneC = 0;
  
  
  if length(varargin) == 2
    if isa(varargin{1}, 'numeric') & isa(varargin{2}, 'numeric')
      m = varargin{1};
      n = varargin{2};
      C = cell(m,n);

      for i = 1:m
	for j = 1:n
	  C{m,n} = tsd;
	end
      end
      doneC = 1;
    end
  end
  
  
  
  if length(varargin) == 1
    if iscell(varargin{1})
      C = varargin{1};
      for i = 1:length(C)
	if ~isa(C{i}, 'tsd')
	  error('arguments must be tsd''s or cell array of tsd''s');
	end
      end
      doneC = 1;
    end
  end
  
  if ~doneC
    C = varargin;
    for i = 1:length(varargin)
      if ~isa(C{i}, 'tsd')
	error('arguments must be array size, tsd''s or cell array of tsd''s');
      end
    end
  end
  
  
  O.C = C;
  
  O = class(O, 'tsdArray');
  
end
  
	
	  
  

  

  

  
  