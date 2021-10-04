function [x,u,traj_cost] = forward_iter(ProblemData, VF)
% This function solves an instance (determined by the given initial state) 
% of optimal control problem using the optimal value functions "VF" given 
% over the discrete state space, via a brute force minimization over the 
% discrete input space.
%
% Inputs: 
%   Data structure containing the problem data (see the block "local variables" below)
%   VF: array of size corresponding to discrete state space for value
%       function
%
% Outputs: 
%   x: vector of size (n_x,T+1) for the controlled state trajectiory 
%   u: vector of size (n_u,T) for the optimal input sequence 
%   traj_cost: a scalar for the total cost of the generated trajectory
%
% NOTE: The extension operation is handeled via the MATALB function 
%                           "griddedInterpolant". 
%      In particular, both the interpolation and extrapolation methods are 
%      set to "linear" (see line 51). However, the user can use other methods
%      that come with MATLAB function "griddedInterpolant". In particular,
%      try to match these with the ones used in the backward value iteration
%      within the VI or CVI algorithms. 
%

%==========================================================================

% local variables (begins) ------------------------------------------------
dyn = ProblemData.Dynamics;

gamma = ProblemData.DiscountFactor;  

cost_x = ProblemData.StateCost;
cost_u = ProblemData.InputCost;
constr_x = ProblemData.StateConstraints;
constr_u = ProblemData.InputConstraints;

X = ProblemData.StateGrid;
U = ProblemData.InputGrid;

x_0 = ProblemData.InitialState;
T = ProblemData.Horizon;

stoch = ProblemData.Stochastic;
if stoch
    W = ProblemData.DiscreteDisturbance;
    P = ProblemData.DisturbancePMF;
    w_t = ProblemData.DisturbanceSequence;
end

% local variables (ends) --------------------------------------------------

% interpolation and extrapolation methods (matching the extension mehtod 
%   used in the backward iteration in VI and CVI)
interpol = 'linear';
extrapol = 'linear';

% constraints
feasibility_t = @(x,u)  max( max(constr_x(x)) , max(constr_u(u)) );

% discrete input cost
disc_cost_u = eval_func_constr(cost_u,U,constr_u); 

n_x = length(x_0); % dimension of the state space
n_u = size(U,1); % dimension of the input space

% allocations
x = zeros(n_x,T+1); % state trajectory
u = zeros(n_u,T); % control input
x(:,1) = x_0;
traj_cost = 0;

%==========================================================================

% forward iteration

for t = 1:T
   
    % minimization over u -------------------------------------------------
    % computing the LERP of the cost-to-go J at (x,u) for u \in U
    x_t = x(:,t);
    if stoch 
        V_at_U = ext_constr_expect(X,VF,U,@(u) (dyn(x_t,u)),constr_x,W,P,interpol,extrapol);
    else
        V_at_U = ext_constr(X,VF,U,@(u) (dyn(x_t,u)),constr_x,interpol,extrapol);
    end
       
    Q = disc_cost_u{1} + gamma*V_at_U;
    [dummy, temp2] = min(Q(:));
    ind_opt = cell(1,n_u);
    [ind_opt{:}] = ind2sub(size(Q),temp2);
    for i=1:n_u
        u(i,t) = U{i}(ind_opt{i});
    end
    % ---------------------------------------------------------------------
    
    % applying the control input
    if stoch
        x(:,t+1) = dyn(x(:,t),u(:,t))+w_t(:,t);
    else
        x(:,t+1) = dyn(x(:,t),u(:,t));
    end
    
    if (feasibility_t(x(:,t),u(:,t)) <=0)
        cost_t = cost_x(x(:,t)) + cost_u(u(:,t));
    else
        cost_t = inf;
    end
    traj_cost = traj_cost + gamma^(t-1) * cost_t;
end

if (constr_x(x(:,T+1)) <=0) 
    cost_T_value = cost_x(x(:,T+1));
else
    cost_T_value = inf;
end
traj_cost = traj_cost + gamma^T * cost_T_value;

