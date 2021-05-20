function [fS,fH]=fusionma(C,S)
% Disclosure: This routine is borrowed from the LLT MATLAB package 
% avialable at "netlib.org/numeralgo/na13"; see also
%      YVES LUCET, FASTER THAN THE FAST LEGENDRE TRANSFORM, THE LINEAR-TIME 
%      LEGENDRE TRANSFORM, NUMERICAL ALGORITHMS, 16 (1997), PP. 171-185 
%
% fusionma(C,S) merges two increasing sequences and gives the resulting 
% slopes fS and indices fH where each slope support the epigraph. All in 
% all, it amounts to finding the first indice i such that
% C(i-1)<=S(j)<C(i).
%
% Usage:
%  [fS,fH]=fusionma(C,S)
%
% Input parameters:
%  C,S - Increasing column vectors. We want to know the conjugate at
%	S(j). C gives the slope of planar input points.
% 
% Output parameter:
%  fS - Slopes supporting the epigraph. This is a subset of S. Column vector.
%  fH - Index at which the slope support the epigraph. Column vector.
%
% Example:
%  X=[-5:0.5:5]'
%  Y=X.^2
%  C=(Y(2:size(Y,1))-Y(1:size(Y,1)-1))./(X(2:size(X,1))-X(1:size(X,1)-1))
%  S=C-0.25*ones(size(C))
%  [fS,fH]=fusionma(C,S)

% The BUILT-IN sort function is faster than looking at each slope with a 
% for loop
[Z, I]=sort([C;S]); 

I=find(I>size(C,1));
J=I(2:size(I,1))-I(1:size(I,1)-1); 
K=J-ones(size(J));
L= cumsum(K)+ones(size(K));

H=[I(1) L'+I(1)-1]';
fS=S;fH=H;
