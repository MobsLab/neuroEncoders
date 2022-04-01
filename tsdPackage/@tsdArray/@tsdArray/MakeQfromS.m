function Q = MakeQfromS(S, DT, varargin)
%
% Q = MakeQfromS(S, DT, parameters)
% 
% 
% INPUTS:
%    S - a cell array of ts objects 
%        (as generated, for example, by LoadSpikes)
%    DT - timestep for ctsd (measured in timestamps!)
%
% OUTPUTS:
%    Q - a ctsd in which the main structure is a |t| x nCells histogram of firing rates
%
% PARAMETERS:
%    T_start: StartTime for the Q matrix, defaults to min(StartTime(S)) class(ts)
%    T_end: EndTime for the Q matrix, defaults to max(EndTime(S)) class(ts)
%    ProgressBar (default 'text'): if 'text', prints "converting n cells: ..."
%                                  if 'graphics', shows display bar
%                                  else shows nothing

% ADR 1998
%  version L5.5
%  status: PROMOTED

% v5.0 30 Oct 1998 time is now first dimension.  INCOMPAT with v4.0.
% v5.1 13 Nov 1998 SCowen found a bug with some cells empty.  Fixed.
% v5.2 18 Nov 1998 Now can create a zero matrix.
% v5.3 19 Nov 1998 ProgresBar flag
% v5.4 21 Nov 1998 fixed [timeIndx T_end] bug
% v5.5 25 Nov 1998 fixed T_end bug

% readapted to tsdArray class, Francesco P. Battaglia 2004
  
  
% --------------------
% Defaults
% --------------------



T_start = inf;
T_end = -inf;
for iC = 1:length(S)
   if ~isempty(Data(S.C{iC})) %class(ts)
      T_start = min(T_start, StartTime(S.C{iC})); %class(ts)
      T_end = max(T_end, EndTime(S.C{iC})); %class(ts)
   end
end

ProgressBar = 'text';
Extract_varargin;

switch ProgressBar
case 'text', flagProgressBar = 1;
case 'graphics', flagProgressBar = 2;
otherwise, flagProgressBar = 0;
end

%--------------------
% Build Q Matrix
%--------------------
spikeTotal =0; cellIndx = []; timeIndx = []; 

nCells = length(S);                                % number of cells
for iC = 1:nCells
   
   if flagProgressBar == 1
      DisplayProgress(iC, nCells, 'UseGraphics', 0, 'Title', ['Converting ', num2str(nCells), ' cells: ']);
   elseif flagProgressBar == 2
      DisplayProgress(iC, nCells, 'UseGraphics', 1, 'Title', ['Converting ', num2str(nCells), ' cells: ']);
   end
      
   if ~isempty(Data(S.C{iC}))  % class(ts)     
      spikeTimes = Restrict(S.C{iC}, T_start, T_end); % class(ts)
      nSpikes = length(Data(spikeTimes)); % class(ts)
      spikeTotal = spikeTotal + nSpikes;
      timeIndx = [timeIndx' Data(spikeTimes)']'; % class(ts)
      b = ones(nSpikes,1).*iC ;  % set element to 1 if there is a
                                 % spike. sparse() does the binning  
                                 
      cellIndx = [cellIndx' b']';
   end; % if ~empty                           
end		% for all cells

timeIndx = round((timeIndx - T_start)/DT)+1; % reset time of first spike in data to zero
endIndx = round((T_end - T_start)/DT)+1;
s = ones(spikeTotal,1);
nTime = max([timeIndx; endIndx]);

if isempty(timeIndx)
   QData = zeros(nTime, nCells); % no spikes, it's a zero-matrix
else
   QData = sparse(timeIndx,cellIndx, s, nTime, nCells); % some matlab functions require full(Q)
end

%--------------------
% Build standard data structure
%--------------------
Q = ctsd(T_start, DT, QData);




