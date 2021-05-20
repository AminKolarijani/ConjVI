function [VF, delta_VF, run_time] = CVI_Alg(ProblemData, CVI_Data)
% This function contains the implementation of the Conjugate VI (CVI)
% algorithm and computes the optimal value function.
% Please see Section 3 of the following article for more details:
%   
%   M.A.S. Kolarijani, C.F. Max, and P. Mohajerin Esfahani (2020), 
%   Fast Approximate Dynamic Programming for Infinite-Horizon 
%   Continuous-State Markov Decision Processes, 
%   preprint arXiv:2102.08880.
%
% Input: 
%   Data structure containing the problem data and CVI configuration (see
%   the block "local variables" below).
%
% Outputs: 
%   VF: array of size corresponding to discrete state space for value
%       function
%   delta_VF: vector of differences (in infinity-norm) of the consecutive 
%             iterates in CVI
%   run_time: vector of size (2,1) for the run-time (in seconds) of the 
%             algorithm, where the first component is compilation time 
%             (particularly, including the time spend for computing the 
%             conjugte of the stage cost numerically), and the second 
%             component is the time spend for value iteration computing 
%             the discrete optimal cost.
%

%==========================================================================

% local variables (begins) ------------------------------------------------
dyn_x = ProblemData.StateDynamics; 
B = ProblemData.InputMatrix;

gamma = ProblemData.DiscountFactor;
 
cost_x = ProblemData.StateCost;
cost_u = ProblemData.InputCost;
constr_x = ProblemData.StateConstraints;
constr_u = ProblemData.InputConstraints;

X = ProblemData.StateGrid;
U = ProblemData.InputGrid;
Ny = ProblemData.StateDualGridSize;
Nz = ProblemData.StateDynamicsGridSize;

alpha_y = CVI_Data.DualGridConstructionAlpha;
dynamic_y = CVI_Data.DynamicDualGridConstruction;

epsilon = CVI_Data.TerminationError;

num_conj_cost = CVI_Data.NumericalConjugateCost;
if num_conj_cost
    Nv = ProblemData.InputDualGridSize;
else
    conj_cost_u = ProblemData.ConjugateInputCost;
end
    
stoch = ProblemData.Stochastic;
if stoch
    W = ProblemData.DiscreteDisturbance;
    P = ProblemData.DisturbancePMF;
    exp_interpol = CVI_Data.ExpectationInterpolationMethod;
    exp_extrapol = CVI_Data.ExpectationExtrapolationMethod;
end
% local variables (ends) --------------------------------------------------    

% allocation 
%J_t = cell(T+1,1); 
run_time = [0; 0];

tic % compilation time (begins)
%==========================================================================

% Computing the conjugate of the input cost numerically
n_x = size(X,1); % dimension of the grid X (and also Y)
n_u = size(U,1); % dimension of the grid U 

% Computing the discrete input cost
tempCi = eval_func_constr(cost_u,U,constr_u); 
disc_cost_u = tempCi{1}; 

% Computing the maximum and minimum of the input cost (to be used for the construction of the grids)
temp_Ci = disc_cost_u(:);  
Max_cost_u = max(temp_Ci(~isinf(temp_Ci)));
Min_cost_u = min(temp_Ci);

% Computing the discrete state cost
tempCs = eval_func_constr(cost_x,X,constr_x); 
disc_cost_x = tempCs{1}; 

if num_conj_cost
    
    % construction of the "uniform" grid V --------------------------------
    Delta_v = slope_range(U,disc_cost_u); % assuming C_i is convex
    V = cell(n_u,1);
    for i = 1:n_u
        temp_v = linspace(Delta_v(i,1),Delta_v(i,2),Nv(i));
        temp_v = [2*temp_v(1)-temp_v(2), temp_v, 2*temp_v(end)-temp_v(end-1)];
        V{i} = unique([temp_v';0]);
    end
    %----------------------------------------------------------------------
    
    disc_conj_cost_u = LLT(U,disc_cost_u,V);
        
end

% construction of the "uniform" grid Z
f_s_X = eval_func(dyn_x,X);
Delta_z = zeros(n_x,2); 
for i = 1:n_x
    Delta_z(i,1) = min(f_s_X{i}(:));
    Delta_z(i,2) = max(f_s_X{i}(:));
end
Z = unif_grid(Delta_z,Nz);

% STATIC construction of the "uniform" grid Y 
Y = cell(n_x,1);
if ~dynamic_y
    
    % Computing the maximum and minimum of the state cost
    temp_Cs = disc_cost_x(:);  
    Max_cost_x = max(temp_Cs(~isinf(temp_Cs)));
    Min_cost_x = min(temp_Cs);
    
    Delta_VF_star_minus_Cs = ( Max_cost_u - Min_cost_u + gamma * (Max_cost_x - Min_cost_x) ) / (1-gamma);
    Delta_y = zeros(n_x,2);
    for i=1:n_x
        Delta_X_i = X{i}(end) - X{i}(1);
        Delta_y(i,2) = alpha_y * Delta_VF_star_minus_Cs / Delta_X_i;
        Delta_y(i,1) = -Delta_y(i,2);
    end

    for i = 1:n_x
        Y{i} = unique([(linspace(Delta_y(i,1),Delta_y(i,2),Ny(i)))';0]);
    end
    
end

%==========================================================================
run_time(1) = run_time(1) + toc; % compilation time (ends)


tic % iteration time (begins)
%==========================================================================
% Recursion: value iteration backward in time

% initialization
VF_plus = disc_cost_x + Min_cost_u;
VF = zeros(size(VF_plus));

delta_VF = max(abs(VF_plus(:) - VF(:)));

% iteration
while (delta_VF(end) >= epsilon) && (length(delta_VF) < 200)
    
    VF = VF_plus;
    VF_holder = VF;
        
    if stoch
        VF = ext_constr_expect(X,VF,X,@(x) x,constr_x,W,P,exp_interpol,exp_extrapol);
    end

    % DYNAMIC construction of the "uniform" grid Y ------------------------
    if dynamic_y
        temp_VF = VF(:);
        Delta_Q = Max_cost_u - Min_cost_u + gamma * ( max(temp_VF(~isinf(temp_VF)))  - min(temp_VF) );
        Delta_y = zeros(n_x,2);
        for i=1:n_x
            Delta_X_i = X{i}(end) - X{i}(1);
            Delta_y(i,2) = alpha_y * Delta_Q / Delta_X_i;
            Delta_y(i,1) = -Delta_y(i,2);
        end
        for i = 1:n_x
            Y{i} = unique([(linspace(Delta_y(i,1),Delta_y(i,2),Ny(i)))';0]);
        end
    end
    %----------------------------------------------------------------------
        
    gVF_conj = LLT(X,gamma*VF,Y);
        
    if num_conj_cost
        phi = ext_constr(V,disc_conj_cost_u,Y,@(y) (-B'*y),@(x) (0),'linear','linear') + gVF_conj;
    else
        temp = eval_func(@(y) conj_cost_u(-B'*y),Y); phi = temp{1}+ gVF_conj;
    end
        
    phi_conj = LLT(Y,phi,Z);
        
    VF_plus_wo_cost_x = ext_constr(Z,phi_conj,X,dyn_x,@(x) (0),'linear','linear');
        
    VF_plus = disc_cost_x + VF_plus_wo_cost_x; 
    
    %temp_error = max(abs(VF_plus(:) - VF(:)));
    temp1 = abs(VF_plus(:) - VF_holder(:));
    temp_error = max(temp1(~isinf(temp1)));
    delta_VF = [delta_VF, temp_error]; 

end

VF = VF_plus;

%==========================================================================
run_time(2) = run_time(2) + toc; % iteration time (ends)



