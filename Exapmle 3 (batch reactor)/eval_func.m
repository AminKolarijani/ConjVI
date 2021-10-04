function F = eval_func(f,X)
% FUNCTION EVALUATION:
% Evalutes the possibly vector valued function given by the handle "f" at 
% the points given by the grid "X".
%
% Usage:
%   F = eval_func(f,X)
%
% Inputs:
%   f - Function handle, takes an n-dimsional column verctor and outputs an m-dimesional column vector
%   X - Cell of size (n,1) of column vectors X{i} for the i-th dimension of the grid X
% 
% Output:
%   F - Cell of size (m,1) of arrays of the dimension corresponding to X for the values of function
%
% Functions called (other than MATLAB routines):
%   None
%
% Example:
% X = cell(2,1); X{1}=[-5:0.5:5]'; X{2}=[-5:0.5:5]';
% f = @(x) (x'*[1 0; 0 1]*x);
% F = eval_func(f,X);
%

n = size(X,1); % dimension of the input argument of the function f
y = f(zeros(n,1));
m = length(y); % dimension of the output argument of the function f

% allocations
ind_max = zeros(1,n);
for i = 1:n
    ind_max(i) = length(X{i});
end

F = cell(m,1);
for j = 1:m
    
    if n==1
        F{j} = zeros(ind_max,1);
    else
        temp_ind = num2cell(ind_max);
        F{j} = zeros(temp_ind{:});
    end
    
end

% evaluations
ind = ones(1,n);

ready = false;
while ~ready
    
    x = zeros(n,1);
    for i=1:n
        x(i) = X{i}(ind(i));
    end
    y = f(x);
    
    for j = 1:m
        temp_ind = num2cell(ind);
        F{j}(temp_ind{:}) = y(j);
    end
    
    ready = true;
    for k = 1:n
        ind(k) = ind(k)+1;
        if ind(k) <= ind_max(k)
            ready = false;
            break;
        end
        ind(k) = 1;
    end

end
