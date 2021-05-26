% This is an illustration of the usage of the developed MATLAB pakage for
%   solving infinite-horizon, discounted cost, optimal control problems 
%   using the devepoled Conjugate VI (CVI) algorithm. Also included is the 
%   implementations of the benchmark VI algorithm. 
%   Please see the follwing article for more details:
%   
%   M.A.S. Kolarijani, C.F. Max, and P. Mohajerin Esfahani (2020), 
%   Fast Approximate Dynamic Programming for Infinite-Horizon 
%   Continuous-State Markov Decision Processes, 
%   preprint arXiv:2102.08880.%   

clc
clear all

%% (1) Problem data (inputs given by the user) 

%--------------------------------------------------------------------------
% Dynamics:
% Deterministic dynamics: f(x,u) = f_s(x) + B*u [input-affine with possibly 
%   nonlinear state dynamics]. Provide the functaion handle for the state  
%   dynamics, and the matrix B for the input dynamics. E.g., for the linear  
%   dyanmics f(x,u) = A*x + B*u:
A = [-.5 2;1 3]; state_dynamics = @(x) (A*x); 
B = [1 .5;1 1]; input_matrix = B;

% Disturbance: x_plus = f(x,u) + w [additive i.i.d. noise]
%
%   NOTE: The current implementation can handle addtitive distrurbance with
%      a FINITE support "W" and a given probability mass function (p.m.f).
%      Then, the expectation with respect to "w" is simply a weighted 
%      average. We note that the algorithm can be easily modified to handle  
%      other forms of approximate expectation (e.g., Monte Carlo). To do 
%      so, one must update the MATLAB function
%                              "ext_constr_expect.m"
%      accordingly. 
%      
%   First, determine if the dynamics is stochastic or deterministic: 
Stoch = true;
%   If "Stoch == true", the user must also provide the finite set of 
%   distrurabnces along with the corresponding probability mass function. 
%   E.g.:
if Stoch
    W1 = [0 .1 -.1]; W = combvec(W1,W1); % discrete set of disturbance (column vectors)
    pmf_W = ones(1,size(W,2))/size(W,2); % disturbance probability mass function
end
%--------------------------------------------------------------------------
% Constraints: provided by function handles as "x_constraints(x) <= 0" for 
%   state constraints, as "u_constraints(u) <= 0" for input constraints.
%   E.g.:
x_constraints = @(x) ([1 0;-1 0; 0 1; 0 -1]*x - ones(4,1)); % state constraints: x \in [-1,1]^2
u_constraints = @(u) ([1 0;-1 0; 0 1; 0 -1]*u - 2*ones(4,1)); % input constraints: u \in [-2,2]^2
%--------------------------------------------------------------------------
% Cost functions: Provide the functions by the proper function handels.  
%   In particular, provide the state-dependent and input-dependent part of 
%   the cost separately. E.g., 
state_cost = @(x) (5*x'*x); % quadratic stage cost (state)
input_cost = @(u) (u'*u); % quadratic stage cost (input)
%--------------------------------------------------------------------------
% Conjugate of input-dependent stage cost: First, determine if the 
%   conjugate is to be computed numerically or is availabe analytically:
Num_Conj = true;
%   If "Num_Conj == false", provide the function handle for the conjugate 
%   of the input-dependent stage cost. E.g., for the quadratic cost given 
%   above, we have (notice that the conjugation takes into account the 
%   input constraints)
if ~Num_Conj
    Delta_u = [-2,2; -2,2]; % i-th row determines the range of i-th input
    conj_input_cost = @(v) (conj_Quad_box(eye(2),Delta_u(:,1),Delta_u(:,2),v));
end
%--------------------------------------------------------------------------
% Discouont factor
gamma = 0.95;
%--------------------------------------------------------------------------
% Optimal control problem instances (initial states and horizon): provide 
%   an initial state for which the optimal control porblem is to be solved. 
%   Also provide the horizon of the forward iteration (a good approximation
%   can be derived by dividing the range of the stage cost by (1-gamma)). 
%   E.g., here we consider T = 200 steps and an initial states randomly 
%   chosen  from the provided state space X = [-1,1]^2:
T = 200; % horizon (a good approximation is )
Delta_x = [-1,1; -1,1]; % i-th row determines the range of i-th state
initial_state = Delta_x(:,1) + (Delta_x(:,2) - Delta_x(:,1)).*rand(size(Delta_x,1),1);

%   If the system is stochastic, then the user must also define a sequence 
%   of disturbances for the forward simulation of the system. The default 
%   mode is to generate a sample sequance using the provided data on the
%   disturbance (i.e., the disturbance set "W" and its p.m.f.)
if Stoch
    ind_w_t = randsample(length(pmf_W),T,true,pmf_W);
    disturb_seq = W(:,ind_w_t);
end 
%--------------------------------------------------------------------------
% Discretization of the state and input (and their dual) spaces: Provide 
%   the GRID-LIKE discretizations of the state and input spaces, as "CELLS".
%   E.g., the state space grid is a cell of size (dimension of x, 1) with
%   each component of the cell inclduing a column vector corresponding to 
%   the discrete points along that dimension of the state space. 

%   NOTE: Here we consider UNIFORM grids for the discritization of the 
%      state and input spaces, but this need not be the case in general.

%   State space discretization (uniform grid)
N_x = [11, 11]; % vector of number of data points in the discretization of state space in EACH dimension
Delta_x = [-1,1; -1,1]; % i-th row determines the range of i-th state
state_grid = unif_grid(Delta_x,N_x); % so the discretization of the state space is a uniform grid of size 11^2 over [-1,1]^2 

%   Input space discretization (uniform grid) - similar to state space discretization described above
N_u = [11, 11]; 
Delta_u = [-2,2; -2,2]; 
input_grid = unif_grid(Delta_u,N_u); 

%   NOTE: The discrete sets X and U must satisfy the feasibility condition, 
%      i.e., for each x in X, there must exist u in U, such that Ax+Bu 
%      satisfies the state constraints. You will recieve a warning if this 
%      is not the case, with an example of a state for which the 
%      feasibility condition is not satisfied.

%   The state dual grid (Y_g):
%     Here, we set the the SIZE of the dual grid Y equal to the size of 
%     the state grid, but this need not be the case in general.
N_y = N_x;
%     Set the value of the coefficeint "alpha" for the construction of the
%     state dual grid; please consult the Section 3.5 of the manuscript 
%     for more details. In particular, take into account the dimension of 
%     the state space in setting the value of alpha. As a rule of thumb, 
%     alpha should be inversly related to the dimension of the state space.
alpha_y = 1;
%     Determine whether Y_g is to be static or constructed dynamically at 
%     eahc iteration; please consult the Section 3.5 of the manuscript for 
%     more details:
dynamic_y = true;

%   The grid (Z_g):
%     Here, we set the the SIZE of the grid Z equal to the size of 
%     the state grid, but this need not be the case in general.
N_z = N_x;

%   The input DUAL grid: If the conjugate of the input stage cost 
%      is to be computed numerically ("Num_Conj == true"), then the user 
%      must determine the size of the input dual grids V. Here, we again  
%      set the the size of the dual grids V equal to the size of the input 
%      grid, but this need not be the case in general.
if Num_Conj
    N_v = N_u;
end

%   NOTE the followings:
%
% (a) The current implementation of the CVI algorithm considers UNIFORM
%     grids for the discretization of the dual spaces. However, in general, 
%     this need not be the case. In order to modify the current 
%     implementation for NON-UNIFORM grid-like discretization of the dual
%     spaces, one needs to modify the correspondig parts of the following
%     MATLAM function
%                               "VI_Conj_Alg.m".
%  
% (b) The vectors "N_x, N_u, N_y, N_z, N_v" determine the number of points  
%     in the grid-like discretization for EACH dimension. So, the grid size 
%     is equal to the product of entries of these vecotrs. Please check the  
%     time complexities of the VI algorithms in the primal and conjugate  
%     domain given in the article. 
%--------------------------------------------------------------------------
% Configuration of the algorithms

%   Determine if you want to also run the (benchmark) VI algorithm, e.g., 
%   in order to compare the performance of the two algorithms
VI_implementation = true;
%   If "VI_implementation = true" and "Stoch == true" (i.e., the system is
%   stochastic), then the user has the option to use the fast approximation  
%   of the stochastic VI opearation. This approximation essentially 
%   replaces the order of expectation and extension operators in stochastic 
%   version of the d-DP algorithm by applying the standard d-DP operator on 
%   J_w(.) = EXPECTAION[J(.+w)]. Use the following to do so:
if VI_implementation && Stoch
    fast_stoch_DP = false;
end

%   NOTE: Considering grid-like discretization of the state space in the
%      d-DP algorithm, the extension operation are handeled via the MATALB
%      function 
%                           "griddedInterpolant". 
%      In particular, both the interpolation and extrapolation methods are 
%      set to "linear" (see below). However, the user can use other methods
%      that come with MATLAB function "griddedInterpolant". 
%      You can use the following lines to modify the methods for 
%      interpolation and extrapolation:
if VI_implementation 
    VI_interpol_mehtod = 'linear';
    VI_extrapol_mehtod = 'linear';
end

%   NOTE: If the dynamics is stochastic, then the d-CDP algorithm also uses
%      extension of the cost function for computing expectation. Once  
%      again, since we use grid-like discretization of the state space the 
%      extension operation uses the MATALB function "griddedinterpolant". 
%      The defualt mode for interpolation and extrapolation methods are 
%      set to "linear" (see below). However, the user can use other methods
%      that come with MATLAB function "griddedinterpolant" to modify them:
if Stoch 
    CVI_interpol_mehtod = 'linear';
    CVI_extrapol_mehtod = 'linear';
end

%   Termination criterion 
epsilon = 0.001; %(Maximum difference between two consecutive iterations 
max_iteration = 1000; % Maximum number of iterations (if the algorithm is not convergent)

%% (2) Problem data (data objects defined for internal use)

ProblemData = struct;

% Deterministic dynamics
ProblemData.StateDynamics = state_dynamics;
ProblemData.InputMatrix = input_matrix;
ProblemData.InputDynamics = @(x) (input_matrix); 
ProblemData.Dynamics = @(x,u) (ProblemData.StateDynamics(x) + ProblemData.InputDynamics(x) * u);

% Disturbance
ProblemData.Stochastic = Stoch;
if Stoch
    ProblemData.DiscreteDisturbance = W; 
    ProblemData.DisturbancePMF = pmf_W; 
end

% Constraints
ProblemData.StateConstraints = x_constraints; 
ProblemData.InputConstraints = u_constraints; 

% Cost functions and conjugates
ProblemData.StateCost = state_cost;
ProblemData.InputCost = input_cost;
if ~Num_Conj 
    ProblemData.ConjugateInputCost = conj_input_cost; 
end

% Discount factor
ProblemData.DiscountFactor = gamma; 

% Control problem instance
ProblemData.Horizon = T;
ProblemData.InitialState = initial_state;
if Stoch
    ProblemData.DisturbanceSequence = disturb_seq; 
end

% Discretizations
ProblemData.StateGridSize = N_x; 
ProblemData.InputGridSize = N_u; 
ProblemData.StateDualGridSize = N_y;
ProblemData.StateDynamicsGridSize = N_z;
if Num_Conj
 ProblemData.InputDualGridSize = N_v; 
end

ProblemData.StateGrid = state_grid;
ProblemData.InputGrid = input_grid;

% Configuration of VI algorithm
if VI_implementation
    VI_Data = struct;
    if Stoch
        VI_Data.FastStochastic = fast_stoch_DP ;
    end
    VI_Data.ExtensionInterpolationMethod = VI_interpol_mehtod;
    VI_Data.ExtensionExtrapolationMethod = VI_extrapol_mehtod;
    VI_Data.TerminationError = epsilon;
    VI_Data.MaxIteration = max_iteration;
end

% Configuration of CVI algorithm
CVI_Data = struct;
CVI_Data.NumericalConjugateCost = Num_Conj;
CVI_Data.DualGridConstructionAlpha = alpha_y;
CVI_Data.DynamicDualGridConstruction = dynamic_y;
CVI_Data.TerminationError = epsilon;
CVI_Data.MaxIteration = max_iteration;
if Stoch 
    CVI_Data.ExpectationInterpolationMethod = CVI_interpol_mehtod;
    CVI_Data.ExpectationExtrapolationMethod = CVI_extrapol_mehtod;
end
 

%% (3) Implementation and Results 
%      NOTE: ALL the outputs are available in the data structure "Result".

%  Output data
Result = struct;

%--------------------------------------------------------------------------

% Feasibility check of discretization scheme: check your command window
%   for possile warnings; the algorithm works even if the discretization of
%   the state and input spaces does not satisfy the feasibility condition
feasibility_check(ProblemData)

%--------------------------------------------------------------------------

% CVI Algorithm
%
%   (1) Value Iteration: The outputs are
%      (a) Result.V_CVI: array of size corresponding to discrete state 
%                        space for value function
%      (b) Result.conv_CVI: vector of differences (in infinity-norm) of 
%                           the consecutive iterates in CVI
%      (c) Result.runtime_CVI: the run-time of the algorithm (in sec)

[Result.V_CVI, Result.conv_CVI, Result.runtime_CVI] = CVI_Alg(ProblemData, CVI_Data);
Result.runtime_CVI = sum(Result.runtime_CVI);

%   (2) Optimal Control Problem (forward iteration for solving the given 
%       instance): The outputs are
%      (a) Result.x_CVI: state trajectiory 
%      (b) Result.u_CVI: input sequence 
%      (c) Result.cost_CVI: the cost of the controlled trajectory 

[Result.x_CVI,Result.u_CVI,Result.cost_CVI] = forward_iter(ProblemData, Result.V_CVI);

%--------------------------------------------------------------------------

if VI_implementation

% VI Algorithm
%
%   (1) Value Iteration: The outputs are
%      (a) Result.V_VI: array of size corresponding to discrete state 
%                        space for value function
%      (b) Result.conv_VI: vector of differences (in infinity-norm) of 
%                           the consecutive iterates in VI
%      (c) Result.runtime_VI: the run-time of the algorithm (in sec)

[Result.V_VI, Result.conv_VI, Result.runtime_VI] = VI_Alg(ProblemData, VI_Data);

%   (2) Optimal Control Problem (forward iteration for solving the given 
%       instance): The outputs are
%      (a) Result.x_VI: state trajectiory 
%      (b) Result.u_VI: input sequence 
%      (c) Result.cost_VI: the cost of the controlled trajectory 

[Result.x_VI,Result.u_VI,Result.cost_VI] = forward_iter(ProblemData, Result.V_VI);

end
