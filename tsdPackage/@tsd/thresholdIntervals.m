function is = thresholdIntervals(tsa, thr, varargin)
% is = thresholdIntervals(tsa, thr, OptionName, OptionValue) intervals in which tsa is
% above (below) threshold
%
% INPUTS:
% tsa: a tsd object
% thr: a threshold value  
% OUTPUTS:
% is: the intervalSet of the times in which tsa is above (below)
% threshold  
% OPTIONS:
% 'Direction': possible values are 'Above' (default) 'Below'
  
% copyright (c) 2004 Francesco P. Battaglia
% This software is released under the GNU GPL
% www.gnu.org/copyleft/gpl.html

 opt_varargin = varargin;
 
 defined_options = dictArray({ { 'Direction', {'Above', { 'char' } } } ...
		   });
 
 getOpt;
 
%rg = timeSpan(tsa);
rg=Range(tsa);

 switch Direction
  case 'Above'
   st = Range(threshold(tsa, thr, 'Crossing', 'Rising', 'InitialPoint', 1));
   en = Range(threshold(tsa, thr, 'Crossing', 'Falling'));
  case 'Below'
   st = Range(threshold(tsa, thr, 'Crossing', 'Falling', 'InitialPoint', 1));
   en = Range(threshold(tsa, thr, 'Crossing', 'Rising'));

  otherwise
   error('Unrecognized option value');
 end
 

   if length(st)<length(en)
	st = [rg(1);st];
   elseif length(st)>length(en)
	en = [en;rg(end)];
   end
 
 is = intervalSet(st, en, '-fixit');
 
 try
     
  dt = diff(Range(tsa));
  mdt = median(dt);
  [miss_ix] = find(dt>3*mdt+eps);
  
  if length(miss_ix)>1
    for i=1:length(miss_ix)
        temp(i,1)=rg(miss_ix(i)-1)+2*mdt;
        temp(i,2)=rg(miss_ix(i)+1)-mdt;
    end
    
    istemp=intervalSet(temp(:,1), temp(:,2));
    is = is-istemp;
  end
  
 end
 
 