function X = unif_grid(Delta_x,N_vec)
% Disclosure: This routine is borrowed from the d-CDp MATLAB package 
% avialable at "https://github.com/AminKolarijani/d-CDP"; see also
%      M.A.S. Kolarijani and P. Mohajerin Esfahani (2020),  
%      Fast approximate dynamic programming for input-affine dynamics,  
%      preprint arXiv:2008.10362.
%
% Generarates a uniform gird "X", with the number of points in each 
% dimension given by the vector N_vec, over the heper-rectangle 
% given by "Delta_x".
%
% Usage:
%  X = unif_grid(Delta_x,N_vec)
%
% Inputs:
%   Delat_x - Matrix of size (n,2), where each row i=1,...,n determines the range of the i-th dimension of the hyperrectangle
%   N_vec - Vector of size (n,1), where N_vec(i) is the number of points in the i-th dimension of the generated grid
% 
% Output:
%   X - Cell of size (n,1) of column vectors X{i} for the i-th dimension of the grid X
%
% Functions called (other than MATLAB routines):
%   None
%
% Example:
%  Delta_x = [-1,1; -1,1];
%  N_vec = 11*ones(2,1);
%  X = unif_grid(Delta_x,N_vec);
%

X = cell(size(Delta_x,1),1);

for i = 1:size(Delta_x,1)
    X{i} = (linspace(Delta_x(i,1),Delta_x(i,2),N_vec(i)))';
end
