function [theta, Rmean, delta, pval] = CircularMean(t)
% [theta, Rmean, delta, pval] = CircularMean(t) ciricualr mean, dispersion
% and significance of a circular datasets
% 
% INPUTS:
% t: a set of angles (in radians) 
% OUTPUT:
% theta: the mean direction
% Rmean: the mean resultant length
% delta: the sample ciricualr dispersion
% pval: the signficance of the mean direction against an uniformity null
% hypothesis (with a Rayleigh test) 
% see Fisher N.I. Analysis of Circular Data p. 30-35

% copyright (c) 2005 Francesco P. Battaglia
% This software is released under the GNU GPL
% www.gnu.org/copyleft/gpl.html

S = sum(sin(t));
C = sum(cos(t));
n = length(t);
theta = atan2(S, C);
Rmean = sqrt(S^2+C^2) / n;

rho2 = sum(cos(2* (t-theta))) / n;

delta = (1 - rho2) / (2 * Rmean^2);

Z = n * Rmean^2;

pval = exp(-Z) * (1 + (2*Z-Z^2)/(4*n) - (24*Z-132*Z^2+76*Z^3-9*Z^4)/(288*n^2));


