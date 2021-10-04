function [SS,Conj]=LLTd(X,Y,S)
% Disclosure: This routine is borrowed from the LLT MATLAB package 
% avialable at "netlib.org/numeralgo/na13"; see also
%      YVES LUCET, FASTER THAN THE FAST LEGENDRE TRANSFORM, THE LINEAR-TIME 
%      LEGENDRE TRANSFORM, NUMERICAL ALGORITHMS, 16 (1997), PP. 171-185 
%
% LLTd computes numerically the discrete Legendre transform of a set of 
% planar points (X(i),Y(i)) at slopes S, i.e., 
%                  Conj(j)=max_{i}[S(j)*X(i)-Y(i)].
% It uses the Linear-time Legendre Transform algorithm to speed up
% computations.
%
% Usage:
%  [SS,Conj]=LLTd(X,Y,S)
%
% Input parameters:
%  X,Y - Column vectors. Usually Y(i)=f(X(i))
%  S - Column vector. We want to know the conjugate at S(j)
%  
% Output parameter:
%  SS - Slopes needed to define the conjugate. This is a column vector. 
%  Conj - numerical values of the conjugate at slopes SS. This is
%	a column vector.
%
% Functions called:
%  bb - Compute the planar convex hull.
%  fusionma - sort two increasing sequences of slopes. This is the core
%  of the LLT algorithm.
%
% Example:
%  X=[-5:0.5:5]'
%  Y=X.^2
%  S=(Y(2:size(Y,1))-Y(1:size(Y,1)-1))./(X(2:size(X,1))-X(1:size(X,1)-1))
%  [SS Conj]=LLTd(X,Y,S)


[XX,YY]=bb(X,Y); % Compute the convex hull (input sensitive algorithm).

h=size(XX,1);
C=(YY(2:h)-YY(1:h-1))./(XX(2:h)-XX(1:h-1)); % Compute the slopes associated 
					    % with the primal points

[SS,H]=fusionma(C,S);

Conj=SS.*XX(H(:,1)) -YY(H(:,1));
