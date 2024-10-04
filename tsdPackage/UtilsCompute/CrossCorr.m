function [C, B] = CrossCorr(t1, t2, binsize, nbins)
% [C, B] = CrossCorr(t1, t2, binsize, nbins)
%
% Cross Correlation of two time series
%
% INPUTS
% t1, t2: arrays containing sorted time series (in the same time unit)
% t1 tref
% t2 tobs
% binsize: size of the bin for the cross correlation histogram
% nbins: number of bins in the histogram
%
% OUTPUTS
% C: the cross correlation histogram
% B: a vector with the time corresponding to the bins (in the same time unit as the input)

% batta 1999
% MEX file
% status: beta

