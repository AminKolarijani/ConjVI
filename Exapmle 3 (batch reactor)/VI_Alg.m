function [VF, delta_VF, run_time] = VI_Alg(ProblemData, VI_Data)
% This function contains the implementation of the VI algorithm and
% computes the optimal value function.
% Please see Section 2 of the extended manuscript for more details.
%   
%
% Inputs: 
%   Data structure containing the problem data and d-DP configuration (see
%   the block "local variables" below).
%
% Outputs: 
%   VF: array of size corresponding to discrete state space for value
%       function
%   delta_VF: vector of differences (in infinity-norm) of the consecutive 
%             iterates in CVI 
%   run_time: a scalar for the run-time (in seconds) of the algorithm
%

%==========================================================================

% local variables (begins) ------------------------------------------------
dyn = ProblemData.Dynamics;

gamma = ProblemData.DiscountFactor;

cost_x = ProblemData.StateCost;
constr_x = ProblemData.StateConstraints;
cost_u = ProblemData.InputCost;
constr_u = ProblemData.InputConstraints;

X = ProblemData.StateGrid;
U = ProblemData.InputGrid;

interpol = VI_Data.ExtensionInterpolationMethod;
extrapol = VI_Data.ExtensionExtrapolationMethod;

stoch = ProblemData.Stochastic;
if stoch
    W = ProblemData.DiscreteDisturbance;
    P = ProblemData.DisturbancePMF;
    fast = VI_Data.FastStochastic;
end

epsilon = VI_Data.TerminationError;
max_iteration = VI_Data.MaxIteration;
% local variables (ends) --------------------------------------------------

tic % algorithm run-time (begins)
%==========================================================================

n_x = size(X,1); % dimension of the state space
n_u = size(U,1); % dimension of the input space
ind_max_x = zeros(1,n_x);
for i = 1:n_x
    ind_max_x(i) = length(X{i});
end 

% discrete costs
disc_cost_u = eval_func_constr(cost_u,U,constr_u);
disc_cost_x = eval_func_constr(cost_x,X,constr_x);

% initialization via terminal cost function
VF_plus = disc_cost_x{1} + min(disc_cost_u{1}(:));
VF = zeros(size(VF_plus));

delta_VF = max(abs(VF_plus(:) - VF(:)));

% iteration
while (delta_VF(end) >= epsilon) && (length(delta_VF) < max_iteration)
    
    VF = VF_plus;
    VF_holder = VF;
    
    if stoch && fast
        VF = ext_constr_expect(X,VF,X,@(x) x,constr_x,W,P,interpol,extrapol);
    end

    ind_x = ones(1,n_x);
    ready = false;
    while ~ready % loop over x \in X
        
        temp_ind = num2cell(ind_x);

        x = zeros(n_x,1);
        for i=1:n_x
            x(i) = X{i}(ind_x(i));
        end
        
        % minimization over u ---------------------------------------------
        
        % computing the LERP of V at (x,u) for u \in U
        if stoch && ~fast 
            VF_at_U = ext_constr_expect(X,VF,U,@(u) (dyn(x,u)),constr_x,W,P,interpol,extrapol);
        else
            VF_at_U = ext_constr(X,VF,U,@(u) (dyn(x,u)),constr_x,interpol,extrapol);
        end
        
        Q = disc_cost_u{1} + gamma * VF_at_U;

        [V_opt, temp2] = min(Q(:));
        % -----------------------------------------------------------------
        
        VF_plus(temp_ind{:}) = V_opt;
        
        ready = true;
        for k = 1:n_x
            ind_x(k) = ind_x(k)+1;
            if ind_x(k) <= ind_max_x(k)
                ready = false;
                break;
            end
            ind_x(k) = 1;
        end

    end
    
    VF_plus = VF_plus + disc_cost_x{1};
    
    %temp_error = max(abs(VF_plus(:) - VF(:)));
    temp1 = abs(VF_plus(:) - VF_holder(:));
    temp_error = max(temp1(~isinf(temp1)));
    
    delta_VF = [delta_VF, temp_error];
    
end

%==========================================================================
run_time = toc; % algorithm run-time (ends)
