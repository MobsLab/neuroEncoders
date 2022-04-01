function O = units(inp, varargin)
% O = units(quantity, unit) constructor for unit 
% 
% quantity is the physical quantity measured by unit (e.g. time, length)
% and unit is the unit of meaasure
  
% copyright (c) 2004 Francesco P. Battaglia 
% This software is released under the GNU GPL
% www.gnu.org/copyleft/gpl.html
  
  if isa(inp, 'units') % copy constructor
    O = inp;
    return
  end
  
  if length(varargin) ~= 1
    error('call like this:  O = units(quantity, unit)');
  end
  
  q = inp;
  u = varargin{1};
  
  
  if (~isa(q, 'char')) | (~isa(u, 'char'))
    error('arguments must be string');
  end
  
  l = get_lookup;
  
  if ~has_key(l, q)
    error('unrecognized quantity');
  end
  
  lt = l{q};
  
  if ~has_key(lt, u)
    error('unrecognized unit');
  end
  
  
  O.quantity = q;
  O.unit = u;
  O.value = lt{u};
  
  O = class(O, 'units');
  
    
    