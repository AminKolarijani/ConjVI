function f_conj = LLT(X,f,S)
% Computes the discrete cojugate of the funcction "f" defined on the grid
% "X" at the slopes given by the grid "S". 
% It uses the univariate Linear-time Legendre Transform (LLTd) algorithm 
% developed by 
%   "Y. Lucet. Faster than the fast Legendre transform, the linear-time 
%       Legendre transform. Numerical Algorithms, 16(2):171-185, 1997,"
% with factorization. 
%
% Usage:
%  f_conj = LLT(X,f,S)
%
% Inputs:
%   X - Cell of size (n_x,1) of column vectors X{i} for the i-th dimension of the grid X
%   f - Array of the dimension corresponding to X for the values of the function 
%   S - Cell of size (n_x,1) of column vectors S{i} for the i-th dimension of the grid S
% 
% Output:
%   f_conj - Array of the dimension corresponding to S for the values of the conjugate function
%
% Functions called (other than MATLAB routines):
%   LLTd - Computes the univariate discrete Legendre transform.
%
% Example:
% X = cell(2,1); X{1}=[-5:0.5:5]'; X{2}=[-5:0.5:5]';
% f = (X{1}*X{2}').^2;
% S = cell(2,1); S{1}=[-200:20:200]'; S{2}=[-200:20:200]';
% f_conj = LLT(X,f,S);
%

n_x = size(X,1); % dimension of the input grid 

if n_x == 1
    [dummy, f_conj] = LLTd(X{1},f,S{1});
    return;
end

% if n_x > 1:

% initialize
F_plus = f;
ind_max = zeros(n_x,1);
for n = 1:n_x
    ind_max(n) = length(X{n});
end

% factirized computations in each dimension
for n = n_x:-1:1
    
    ind = ones(1,n_x);
    ind_set = [1:n-1,n+1:n_x];
    ind_max(n) = length(S{n});
    
    temp_ind = num2cell(ind_max);
    F = zeros(temp_ind{:});
    
    ready = false;
    
    while ~ready
        
        temp_1 = zeros(length(X{n}),1);
        for i = 1:length(X{n})
            ind(n) = i;
            temp_ind = num2cell(ind);
            temp_1(i) = F_plus(temp_ind{:});
        end
        
        if n==n_x
            [dummy temp_2] = LLTd(X{n},temp_1,S{n});
        else
            [dummy temp_2] = LLTd(X{n},-temp_1,S{n});
        end
        
        for j = 1:length(S{n})
            ind(n) = j;
            temp_ind = num2cell(ind);
            F(temp_ind{:}) = temp_2(j);
        end
        
        ready = true;
        
        for k = ind_set
            ind(k) = ind(k)+1;
            if ind(k) <= ind_max(k)
                ready = false;
                break;
            end
            ind(k) = 1;
        end
        
    end
    
    F_plus = F;
    
end

f_conj = F_plus;
