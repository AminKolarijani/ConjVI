function F_h = eval_func_constr(f,X,h)
% Disclosure: This routine is borrowed from the d-CDp MATLAB package 
% avialable at "https://github.com/AminKolarijani/d-CDP"; see also
%      M.A.S. Kolarijani and P. Mohajerin Esfahani (2020),  
%      Fast approximate dynamic programming for input-affine dynamics,  
%      preprint arXiv:2008.10362.
%
% CONSTRAINED FUNCTION EVALUATION
% Evalutes the function given by the handle "f" at points given by the grid 
% "X", while checking the constraints "h(x)<=0" [component-wise], given by 
% the other function handle "h".
%
% Usage:
%   F = eval_func_constr(f,X,h)
%
% Inputs:
%   f - Function handle, takes an n-dimsional column verctor and outputs an m-dimesional column vector
%   X - Cell of size (n,1) of column vectors X{i} for the i-th dimension of the grid X
%   h - Function handle, takes an n-dimsional column verctor and outputs a column vector
% 
% Output:
%   F_h - Cell of size (m,1) of arrays of the dimension corresponding to X for the values of function
%
% Functions called (other than MATLAB routines):
%   None
%
% Example:
% X = cell(2,1); X{1}=[-5:0.5:5]'; X{2}=[-5:0.5:5]';
% f = @(x) (x'*[1 0; 0 1]*x);
% h = @(x) (x(1)+x(2)-8);
% F_h = eval_func_constr(f,X,h);
%

n = size(X,1); % dimension of the input argument of the function f
y = f(zeros(n,1));
m = length(y); % dimension of the output argument of the function f


% allocations
ind_max_x = zeros(1,n);
for i = 1:n
    ind_max_x(i) = length(X{i});
end

F_h = cell(m,1);
for j = 1:m
    if n==1
        F_h{j} = zeros(ind_max_x,1);
    else
        temp_ind = num2cell(ind_max_x);
        F_h{j} = zeros(temp_ind{:});
    end
end

% evaluations
ind_x = ones(1,n);
ready = false;
while ~ready
    
    x = zeros(n,1);
    for i=1:n
        x(i) = X{i}(ind_x(i));
    end
    y = f(x);
    
    for j = 1:m
        temp_ind = num2cell(ind_x);
        if h(x)<=0 % checking the constraints
            F_h{j}(temp_ind{:}) = y(j);
        else
            F_h{j}(temp_ind{:}) = inf;
        end
    end
    
    ready = true;
    for k = 1:n
        ind_x(k) = ind_x(k)+1;
        if ind_x(k) <= ind_max_x(k)
            ready = false;
            break;
        end
        ind_x(k) = 1;
    end

end

