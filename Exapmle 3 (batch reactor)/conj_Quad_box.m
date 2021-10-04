function g_conj = conj_Quad_box(R,u_min,u_max,v)
% Computes the conjugate of the extended real-valued function 
%              g(u) = u'*R*u, [weighted L_2 norm]
% at the point "v", where 
%   (a) the effective domain of "g" is the hyper-rectangle given by 
%                         u_min <= u <= u_max;
%   (b) the effective domain of "g" contains the origin;
%   (c) the weight matrix "R" is positive definite.
%
% Usage:
%   g_conj = conj_Quad_box(R,u_min,u_max,v)
%
% Inputs:
%   R - m by m matrix, where m is the dimension of the variables u and v.
%   u_min - m-dimsional column verctor determining the lower bounds of the 
%           effective domain of g 
%   u_max - m-dimsional column verctor determining the upper bounds of the 
%           effective domain of g
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
% g_conj = conj_Quad_box(eye(2),[-2;2],[-2;2],[3;1]);
%

u = .5*(R\v);

if (u >= u_min) & (u <= u_max)
    g_conj = .25*v'*(R\v);
else
    for i = 1:length(v)
        u(i) = max( u_min(i), min(u(i),u_max(i)) );
    end
    g_conj = u'*v - u'*R*u;    
end