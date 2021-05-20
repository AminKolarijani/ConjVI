function g_conj = conj_ExpL1_box(u_min,u_max,v)
% Disclosure: This routine is borrowed from the d-CDp MATLAB package 
% avialable at "https://github.com/AminKolarijani/d-CDP"; see also
%      M.A.S. Kolarijani and P. Mohajerin Esfahani (2020),  
%      Fast approximate dynamic programming for input-affine dynamics,  
%      preprint arXiv:2008.10362.
%
% Computes the conjugate of the extended real-valued function 
%              g(u) = sum_i {exp(abs(u_i)) - m}, 
% at the point "v", where
%   (a) "m" is the dimension of the variables "u" and "m"; 
%   (b) the effective domain of "g" is the hyper-rectangle given by 
%                         u_min <= u <= u_max;
%   (c) the effective domain of "g" contains the origin.
%
% Usage:
%   g_conj = conj_ExpL1_box(u_min,u_max,v);
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
% g_conj = conj_ExpL1_box([-2;2],[-2;2],[3;1]);
%

g_conj_i = zeros(size(v));

for i = 1:length(v)
    
    if abs(v(i)) > 1 
        u_i = sign(v(i)) * log(abs(v(i)));
        if (u_i >= u_min(i)) && (u_i <= u_max(i))
            g_conj_i(i) = abs(v(i)) *(log(abs(v(i))) - 1)+1;
        else
            u_i = max( u_min(i), min(u_i,u_max(i)) );
            g_conj_i(i) = u_i * v(i) - exp(abs(u_i)) + 1;
        end
    end
end

g_conj = sum(g_conj_i);

    