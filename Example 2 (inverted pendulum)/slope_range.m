function Delta_s = slope_range(X,J)
% Disclosure: This routine is borrowed from the d-CDP MATLAB package 
% avialable at "https://github.com/AminKolarijani/d-CDP"; see also
%      M.A.S. Kolarijani and P. Mohajerin Esfahani (2020),  
%      Fast approximate dynamic programming for input-affine dynamics,  
%      preprint arXiv:2008.10362.
%
% Uses the discrete data points (X,J(X)) of the "extended real-valued
% convex" function "J" over the grid "X",to compute the ragnge of slopes 
% (differences) of the discrete function J in each dimension corresponding 
% to the grid "X".
%
% Usage:
%   Delta_s = slope_range(X,J)
%
% Inputs:
%   X - Cell of size (n,1) of column vectors X{i} for the i-th dimension of the grid X
%   J - Array of the dimension corresponding to X for the values of the function
% 
% Output:
%   Delat_s - Matrix of size (n,2), where each row i=1,...,n determines the range of the slopes of "J" in the i-th dimension
%
% Functions called (other than MATLAB routines):
%   None
%
% Example:
% X = cell(2,1); X{1}=[-5:0.5:5]'; X{2}=[-5:1:5]';
% J = (X{1}*X{2}').^2;
% Delta_s = slope_range(X,J);
%

n_x = size(X,1); % dimension of the input grid 
Delta_s = zeros(n_x,2);

% J convex-extensible (method == 'convex')
ind_max_x = zeros(1,n_x);
for i = 1:n_x
    ind_max_x(i) = length(X{i});
end

for i = 1:n_x
    
    % computing the FFD and LBD of J along each dimension
    ind_x = ones(1,n_x);
    ind_set_x = [1:i-1,i+1:n_x];
    
    FFDi_J = []; % First Forward Difference of J along i-th dimension
    LBFi_J = []; % Last Backward Difference of J along i-th dimension

    ready = false;
    while ~ready
        
        for j = 1:length(X{i})
            ind_1 = ind_x; ind_1(i) = j; temp_ind_1 = num2cell(ind_1);
            if ~isinf(J(temp_ind_1{:}))
                if (j < length(X{i})) 
                    ind_2 = ind_x; ind_2(i) = j+1; temp_ind_2 = num2cell(ind_2);
                    if ~isinf(J(temp_ind_2{:}))
                        FFDi_J = [FFDi_J; (J(temp_ind_2{:})-J(temp_ind_1{:}))/(X{i}(j+1)-X{i}(j))];
                    else
                        FFDi_J = [FFDi_J; 0];
                    end
                else 
                    FFDi_J = [FFDi_J; 0];
                end
                break;
            end
        end
        
        for j = length(X{i}):-1:1
            ind_1 = ind_x; ind_1(i) = j; temp_ind_1 = num2cell(ind_1);
            if ~isinf(J(temp_ind_1{:}))
                if (j > 1) 
                    ind_2 = ind_x; ind_2(i) = j-1; temp_ind_2 = num2cell(ind_2);
                    if ~isinf(J(temp_ind_2{:}))
                        LBFi_J = [LBFi_J; (J(temp_ind_1{:})-J(temp_ind_2{:}))/(X{i}(j)-X{i}(j-1))];
                    else
                        LBFi_J = [LBFi_J; 0];
                    end
                else 
                    LBFi_J = [LBFi_J; 0];
                end
                break;
            end
        end
        
        ready = true;
        for k = ind_set_x
            ind_x(k) = ind_x(k)+1;
            if ind_x(k) <= ind_max_x(k)
                ready = false;
                break;
            end
            ind_x(k) = 1;
        end

    end
    
    Delta_s(i,1) = min(FFDi_J);
    Delta_s(i,2) = max(LBFi_J);
      
end