function f_g_y = ext_constr(X,f,Y,g,h,interpol,extrapol)
% CONSTRAINED EXTENSION:
% Uses the discrete data points (X,f(X)) of the function "f" over the grid "X", 
% to approximate the values "f(g(Y))" over the points "g(Y)" given by the 
% function handle "g" and the grid "Y". It also checks the constraints 
% "h(g(Y)) <=0", given by the function handle "h". The routine uses the 
% extension methods specified by the "interpol" and "extrapol" inpput arguments.
%
% Usage:
%   f_g_y = ext_constr(X,f,Y,g,h,interpol,extrapol)
%
% Inputs:
%   X - Cell of size (n_x,1) of column vectors X{i} for the i-th dimension of the grid X
%   f - Array of the dimension corresponding to X for the values of the function
%   Y - Cell of size (n_y,1) of column vectors Y{j} for the j-th dimension of the grid Y
%   g - Function handle, takes an n_y-dimsional column verctor (y) and outputs an n_x-dimesional column vector (x)
%   h - Function handle, takes an n_x-dimsional column verctor (x) and outputs a column vector
%   interpol -  Determines the interpolation method (see the MATLAB routine griddedInterpolant)  
%   extrapol -  Determines the extrapolation method (see the MATLAB routine griddedInterpolant)
% 
% Output:
%   f_g_y - Array of the dimension corresponding to Y for the values f(g(y)) for y \in Y satisfying the constraint h(g(y))<=0.
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
% f_g_y = ext_constr(X,f,Y,g,h,interpol,extrapol);
%

F = griddedInterpolant(X,f,interpol,extrapol);

n_y = size(Y,1); % dimension of the output grid

% allocations
ind_max_y = zeros(1,n_y);
for i = 1:n_y
    ind_max_y(i) = length(Y{i});
end

if n_y==1
        f_g_y = zeros(ind_max_y,1);
else
    temp_ind = num2cell(ind_max_y);
    f_g_y = zeros(temp_ind{:});
end

% evaluations
ind_y = ones(1,n_y);
ready = false;
while ~ready
    
    y = zeros(n_y,1);
    for i=1:n_y
        y(i) = Y{i}(ind_y(i));
    end
    temp_ind = num2cell(ind_y);
    if h(g(y))<=0 % checking the constraints
        f_g_y(temp_ind{:}) = F(g(y)');
    else
        f_g_y(temp_ind{:}) = inf;
    end
    
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