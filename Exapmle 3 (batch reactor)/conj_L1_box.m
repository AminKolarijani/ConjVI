function g_conj = conj_L1_box(u_min,u_max,v)
% Computes the conjugate of the extended real-valued function 
%              g(u) = sum_i {abs(u_i)}, [L_1 norm]
% at the point "v", where 
%   (a) the effective domain of "g" is the hyper-rectangle given by 
%                         u_min <= u <= u_max;
%   (b) the effective domain of "g" contains the origin.
%
% Usage:
%   g_conj = conj_L1_box(u_min,u_max,v);
%
% Inputs:
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
% g_conj = conj_L1_box([-2;2],[-2;2],[3;1]);
%

g_conj_i = zeros(size(v));

for i = 1:length(v)
    
    if v(i) > 1
        g_conj_i(i) = u_max(i) * v(i) - u_max(i);
    elseif v(i) < -1
        g_conj_i(i) = u_min(i) * v(i) + u_min(i);
    end
end

g_conj = sum(g_conj_i);