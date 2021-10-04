function E_f_g_y = ext_constr_expect(X,f,Y,g,h,W,P,interpol,extrapol)
% Disclosure: This routine is borrowed from the d-CDP MATLAB package 
% avialable at "https://github.com/AminKolarijani/d-CDP"; see also
%      M.A.S. Kolarijani and P. Mohajerin Esfahani (2020),  
%      Fast approximate dynamic programming for input-affine dynamics,  
%      preprint arXiv:2008.10362.
%
% EXPECTATION USING CONSTRAINED EXTENION:
% An extension of the routine "ext_constr.m" -- computes the expected value 
%                    E_w[f(g(y+w))] for "y \in Y", 
% where "w \in W" are the disturbances, and "P" is the corresponding pmf. 
%
% Usage:
%   E_f_g_y = ext_constr_exp(X,f,Y,g,h,W,P,interpol,extrapol)
%
% Inputs:
%   X - Cell of size (n_x,1) of column vectors X{i} for the i-th dimension of the grid X
%   f - Array of the dimension corresponding to X for the values of the function
%   Y - Cell of size (n_y,1) of column vectors Y{i} for the i-th dimension of the grid Y
%   g - Function handle, takes an n_y-dimsional column verctor (y) and outputs an n_x-dimesional column vector (x)
%   h - Function handle, takes an n_x-dimsional column verctor (x) and outputs a column vector
%   W - Matirx of size (n_x,n_w) of column vectors W(:,i) of the distrubances 
%   P - Row vector of length n_w for the probabilities of the distrubances
%   interpol -  Determines the interpolation method (see the MATLAB routine griddedInterpolant)  
%   extrapol -  Determines the extrapolation method (see the MATLAB routine griddedInterpolant))
% 
% Output:
%   E_f_g_y - Array of the dimension corresponding to Y for the values E_w[f(g(y+w))] for y \in Y
%
% Functions called (other than MATLAB routines):
%   None
%
% Example:
% X = cell(2,1); X{1}=[-5:0.5:5]'; X{2}=[-5:1:5]';
% Y = cell(2,1); Y{1}=[-1:1:6]'; Y{2}=[-1:1:6]';
% f = (X{1}*X{2}').^2;
% g = @(y) ([1 1; 0 1]*y);
% h = @(x) (x(1)+x(2)-8);
% interpol = 'linear';
% extrapol = 'linear';
% W1 = [0 .1 -.1];
% W = combvec(W1,W1);
% P = ones(1,size(W,2))/size(W,2);
% E_f_g_y = ext_constr_exp(X,f,Y,g,h,W,P,interpol,extrapol);
%

F = griddedInterpolant(X,f,interpol,extrapol);


n_y = size(Y,1); % dimension of the output grid
n_w = length(P); % size of the disturbance set W

% allocations
ind_max_y = zeros(1,n_y);
for i = 1:n_y
    ind_max_y(i) = length(Y{i});
end

if n_y==1
        E_f_g_y = zeros(ind_max_y,1);
else
    temp_ind = num2cell(ind_max_y);
    E_f_g_y = zeros(temp_ind{:});
end

% evaluations
ind_y = ones(1,n_y);
ready = false;
while ~ready
    
    y = zeros(n_y,1);
    for i=1:n_y
        y(i) = Y{i}(ind_y(i));
    end
    f_g_y_w = zeros(size(P));
    for i = 1:n_w
        
        % soft constraints
        if h(g(y)+W(:,i))<=0 % checking the constraints
            f_g_y_w(i) = F((g(y)+W(:,i))'); % computing the function value 
            % using the extension for disturbances that respect the constriants
        else
            f_g_y_w(i) = inf; % if the disturbance does not respect
            % the constriants, we set the value to infinity.
        end
        
%         % hard constraints
%         if h(g(y)+W(:,i))<=0 % checking the constraints
%             f_g_y_w(i) = F((g(y)+W(:,i))'); % computing the function value 
%             % using the extension for disturbances that respect the constriants
%         else
%             f_g_y_w(i) = F(g(y)'); % if the disturbance does not respect
%             % the constriants, we disgard them (hard constraints).
%         end
        
    end
    
    temp_ind = num2cell(ind_y);
    E_f_g_y(temp_ind{:}) = sum(P.*f_g_y_w);
    
    ready = true;
    for k = 1:n_y
        ind_y(k) = ind_y(k)+1;
        if ind_y(k) <= ind_max_y(k)
            ready = false;
            break;
        end
        ind_y(k) = 1;
    end

end