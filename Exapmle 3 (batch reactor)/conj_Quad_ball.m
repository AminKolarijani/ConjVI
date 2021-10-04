function g_conj = conj_Quad_ball(R,r,v)
% Computes the conjugate of the extended real-valued function 
%              g(u) = u'*R*u, [weighted L_2 norm]
% at the point "v", where 
%   (a) the effective domain of "g" is ball of radius "r", centered at origin ;
%   (b) the weight matrix "R" is positive definite.
%
% Usage:
%   g_conj = conj_Quad_ball(R,r,v)
%
% Inputs:
%   R - m by m matrix, where m is the dimension of the variables u and v. 
%   r- the radius of the effective domain of g
%   v - m-dimsional column verctor at which the conjugate of g is to be
%       evaluated
% 
% Output:
%   g_conj - the values (real number) of the conjugate of g at the point v  
%
% Functions called (other than MATLAB routines):
%   None
%
% Example:
% g_conj = conj_Quad_ball(eye(2),2,[1;-1]);
%

u = .5*(R\v);
if (sum(u.*u))^(.5) <= r
    g_conj = .25*v'*(R\v);
else
    u = r*u/(sum(u.*u))^(.5);
    g_conj = u'*v - u'*R*u;    
end