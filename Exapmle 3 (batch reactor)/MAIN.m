% This file is an example of implementation of the CVI and VI algorithms 
% for solving the infinite-horizon optimal control problem for an 
% unstable batch reactor. 
% Please also see Section 4.3 of the following article for more details:
%   
%   M.A.S. Kolarijani, C.F. Max, and P. Mohajerin Esfahani (2021), 
%   Fast Approximate Dynamic Programming for Infinite-Horizon 
%   Continuous-State Markov Decision Processes, 
%   preprint arXiv:2102.08880.
%

clc
clear all

%% Optimal control problem instances (initial states and horizon) 
T = 100; % horizon
NumInstances = 100; % the number of instances (initial states)

% Initial states
initial_state_set = cell(NumInstances,1); % the set of initial states (allocation)
Delta_x = repmat([-1,1],4,1); % i-th row determines the range of i-th state
for i_Inst = 1:NumInstances
    initial_state_set{i_Inst} = Delta_x(:,1) + (Delta_x(:,2) - Delta_x(:,1)).*rand(size(Delta_x,1),1);
end

% Size of the grids for discretization
N_set = [5 7 11 15 21 25]; % grid sizes in each dimension

i_N = 1;
save GS_IS_Data

%% Implementation of algorithms and solving the problem instances

while i_N <= length(N_set) % iteraton over grid sizes
    
    clear all
    
    % Optimal control problem instances (initial states and horizon)
    load GS_IS_Data
    
    %======================================================================
    
    % (1) Problem data (inputs given by the user)

    % Dynamics: f(x,u) = Ax+Bu (linear dynamics)
    A = [1.08, -0.05, 0.29, -0.24;...
        -0.03, 0.81, 0, 0.03;...
        0.04, 0.19, 0.73, 0.24;...
        0.00, 0.19, 0.05, 0.91]; 
    state_dynamics = @(x) (A*x); 
    B = [0, -0.02;...
        0.26, 0;...
        0.08, -0.13;...
        0.08, 0]; 
    input_matrix = B;
    
    d_x = 4;
    d_u = 2;
    
    % Constraints
    x_constraints = @(x) ([eye(d_x); -eye(d_x)]*x - 2*ones(2*d_x,1)); % state constraints: x \in [-2,2]^4
    u_constraints = @(u) ([eye(d_u); -eye(d_u)]*u - 2*ones(2*d_u,1)); % input constraints: u \in [-2,2]^2

    % Cost functions
    state_cost = @(x) (2*x'*x); % quadratic stage cost (state)
    input_cost = @(u) (u'*u); % quadratic input cost (input)
%     state_cost = @(x) (-(4/1.01)+sum(1./(1.01- abs(x)))); % 
%     input_cost = @(u) (-(2/2.01)+sum(1./(2.01- abs(u)))); % 

    % Conjugate of input-dependent stage cost
    Delta_u = repmat([-2,2],2,1); % i-th row determines the range of i-th input
    
    % Discouont factor
    gamma = 0.95;

    % Discretization of the state and input (and their dual) spaces    
    %   State space discretization (uniform grid)
    N_x = N_set(i_N)*ones(1,d_x); % vector of number of data points in the discretization of state space in EACH dimension
    % Delta_x = repmat([-2,2],4,1); % i-th row determines the range of i-th state
    state_grid = unif_grid(Delta_x,N_x); % so the discretization of the state space is a uniform grid of size 11^2 over [-1,1]^2 
    %   Input space discretization (uniform grid) - similar to state space discretization described above
    N_u = N_set(i_N)*ones(1,d_u);
    %Delta_u = repmat([-2,2],2,1); 
    input_grid = unif_grid(Delta_u,N_u); 
    %   The state dual grid (Y_g):
    N_y = N_x; 
    %   The grid (Z_g):
    N_z = N_x;
    %   The input dual grid (V_g):
    N_v = N_u;
    %   Set the value of the coefficeint "alpha" 
    alpha_y = 1;
    
    % Configuration of VI and CVI algorithms
    VI_implementation = true;

    VI_interpol_mehtod = 'linear';
    VI_extrapol_mehtod = 'linear';
    
    epsilon = 0.001; % Termination bound
    max_iteration = 1000; % Maximum number of iterations (if the algorithm is not convergent)
    %======================================================================
    
    % (2) Problem data (data objects defined for internal use)

    ProblemData = struct;

    % Dynamics
    ProblemData.StateDynamics = state_dynamics;
    ProblemData.InputMatrix = input_matrix;
    ProblemData.InputDynamics = @(x) (input_matrix); 
    ProblemData.Dynamics = @(x,u) (ProblemData.StateDynamics(x) + ProblemData.InputDynamics(x) * u);
    
    % Constraints
    ProblemData.StateConstraints = x_constraints; 
    ProblemData.InputConstraints = u_constraints; 

    % Cost functions and conjugates
    ProblemData.StateCost = state_cost;
    ProblemData.InputCost = input_cost;

    % Discount factor
    ProblemData.DiscountFactor = gamma; 

    % Control problem instance
    ProblemData.Horizon = T;

    % Discretizations
    ProblemData.StateGridSize = N_x; 
    ProblemData.InputGridSize = N_u; 
    ProblemData.StateDualGridSize = N_y;
    ProblemData.StateDynamicsGridSize = N_z;
    ProblemData.InputDualGridSize = N_v; 
    
    ProblemData.StateGrid = state_grid;
    ProblemData.InputGrid = input_grid;
    
    % Configuration of VI and CVI algorithms
    VI_Data = struct;
    VI_Data.ExtensionInterpolationMethod = VI_interpol_mehtod;
    VI_Data.ExtensionExtrapolationMethod = VI_extrapol_mehtod;
    VI_Data.TerminationError = epsilon;
    VI_Data.MaxIteration = max_iteration;

    CVI_Data = struct;
    CVI_Data.TerminationError = epsilon;
    CVI_Data.MaxIteration = max_iteration;
    CVI_Data.ExpectationInterpolationMethod = CVI_interpol_mehtod;
    CVI_Data.ExpectationExtrapolationMethod = CVI_extrapol_mehtod;
    
    %======================================================================
    
%%    % Feasibility check of discretization scheme
    ProblemData.Stochastic = false;
    feasibility_check(ProblemData)
    
    %======================================================================
%%    % (3) Implementation and Results
    
    %  Output data
    Result = struct;
    Result.InitialState = initial_state_set;
    
    %----------------------------------------------------------------------
    % CVI Algorithm 
    
    ProblemData.Stochastic = false; % Stochastic dynamcis
    CVI_Data.NumericalConjugateCost = true; % numerical computation of the conjugate of input cost
    CVI_Data.DualGridConstructionAlpha = 1; % the coefficient alpha for construction of Y_g
    CVI_Data.DynamicDualGridConstruction = false; % dynamic construction of Y_g
    
    [temp1, temp2, temp3] = CVI_Alg(ProblemData, CVI_Data);
    Result.V_CVI = temp1;
    Result.conv_CVI = temp2;
    Result.rt_CVI = sum(temp3);

    for i_Inst = 1:NumInstances % Iteraton over control problem instances

        ProblemData.InitialState = Result.InitialState{i_Inst}; 
        ProblemData.DisturbanceSequence = Result.DisturbanceSequence{i_Inst}; 

        [dummy1, dummy2, temp4] = forward_iter(ProblemData, temp1);
        Result.tc_CVI(i_Inst) = temp4;
    end
    
    %----------------------------------------------------------------------
    % CVI Algorithm (dynamic contruction of Y_g)
    
    ProblemData.Stochastic = false; % Stochastic dynamcis
    CVI_Data.NumericalConjugateCost = true; % numerical computation of the conjugate of input cost
    CVI_Data.DualGridConstructionAlpha = 1; % the coefficient alpha for construction of Y_g
    CVI_Data.DynamicDualGridConstruction = true; % dynamic construction of Y_g
    
    [temp5, temp6, temp7] = CVI_Alg(ProblemData, CVI_Data);
    Result.V_CVI_d = temp5;
    Result.conv_CVI_d = temp6;
    Result.rt_CVI_d = sum(temp7);

    for i_Inst = 1:NumInstances % Iteraton over control problem instances

        ProblemData.InitialState = Result.InitialState{i_Inst};
        ProblemData.DisturbanceSequence = Result.DisturbanceSequence{i_Inst}; 

        [dummy3, dummy4, temp8] = forward_iter(ProblemData, temp5);
        Result.tc_CVI_d(i_Inst) = temp8;
    end

    %----------------------------------------------------------------------
    % VI Algorithm
    
    ProblemData.Stochastic = false; % Stochastic dynamcis
    
    [temp13, temp14, temp15] = VI_Alg(ProblemData, VI_Data);
    Result.V_VI = temp13;
    Result.conv_VI = temp14;
    Result.rt_VI = temp15;

    for i_Inst = 1:NumInstances % Iteraton over control problem instances
        
        ProblemData.InitialState = Result.InitialState{i_Inst};
        ProblemData.DisturbanceSequence = Result.DisturbanceSequence{i_Inst};
        
        [dummy7, dummy8, temp16] = forward_iter(ProblemData, temp13);
        Result.tc_VI(i_Inst) = temp16;
    end
    
  
%----------------------------------------------------------------------

% Saving the results for each grid size 
    str = sprintf('Ex_3_Data_N%2.0f',N_set(i_N));
    save(str,'ProblemData','VI_Data','CVI_Data','Result')
    
    % Next grid size
    i_N = i_N+1
    save('GS_IS_Data','i_N','-append')

end

%% Analysis of results

gamma = 0.95; % discount factor
N_set = [5 7 11 15 21 25]; % grid sizes 5^4, ...

% Convergence rate of VI and CVI for largest grid size
i_N = length(N_set);
N = N_set(i_N);
str = sprintf('Ex_3_Data_N%2.0f',N);
load(str)

Convergence_VI = Result.conv_VI(2:end);
Convergence_CVI = Result.conv_CVI(2:end);  
Convergence_CVI_d = Result.conv_CVI_d(2:end); % CVI with dynamic Y_g

gamma_rate = zeros(1,length(Convergence_VI));
gamma_rate(1) = Convergence_CVI(1);
for i=1:length(Convergence_VI)
    gamma_rate(i+1) = gamma*gamma_rate(i);
end

h1 = figure;
semilogy(1:length(Convergence_VI),Convergence_VI,'Color',[.0 .45 .74],'LineStyle','-','Marker','none','LineWidth',1.5)
hold on
semilogy(1:length(Convergence_CVI),Convergence_CVI,'Color',[0 0.5 0],'LineStyle','-.','Marker','none','LineWidth',1.5)
semilogy(1:length(Convergence_CVI_d),Convergence_CVI_d,'Color',[.85 .33 .1],'LineStyle','--','Marker','none','LineWidth',1.5)
semilogy(1:length(gamma_rate),gamma_rate,'Color','k','LineStyle',':','Marker','none','LineWidth',1.5)
hold off
ylabel('$\Vert J - J_+ \Vert_{\infty}$','FontSize',10,'Interpreter','latex')
xlabel('Iteration, $k$','FontSize',10,'Interpreter','latex')
txt1 = 'VI';
txt2 = 'CVI';
txt3 = 'CVI-d';
legend({txt1, txt2, txt3},'Interpreter','latex','Orientation','vertical','Location','northeast','FontSize',10)
ax = gca;
ax.Box = 'on';
grid on

% %==========================================================================
% Running time of VI and CVI for different grid sizes
RunTime_VI = zeros(1,length(N_set));
RunTime_CVI = zeros(1,length(N_set)); 
RunTime_CVI_d = zeros(1,length(N_set));  % CVI with dynamic Y_g

for i_N = 1:length(N_set)
    
    N = N_set(i_N);
    str = sprintf('Ex_3_Data_N%2.0f',N);
    load(str)
    
    RunTime_VI(i_N) = Result.rt_VI;
    RunTime_CVI(i_N) = Result.rt_CVI;
    RunTime_CVI_d(i_N) = Result.rt_CVI_d;
    
end

h2 = figure;
N_vec = (N_set.^6);
loglog(N_vec,RunTime_VI,'Color',[.0 .45 .74],'LineStyle','-','Marker','s','LineWidth',1)
hold on
loglog(N_vec,RunTime_CVI,'Color',[0 0.5 0],'LineStyle','-','Marker','d','LineWidth',1)
loglog(N_vec,RunTime_CVI_d,'Color',[.85 .33 .1],'LineStyle','-','Marker','o','LineWidth',1)
hold off
ylabel('Time (sec)','FontSize',10,'Interpreter','latex')
xlabel('Grid size, $X\times U$','FontSize',10,'Interpreter','latex')
txt1 = 'VI';
txt2 = 'CVI';
txt3 = 'CVI-d';
legend({txt1, txt2, txt3},'Interpreter','latex','Orientation','vertical','Location','northwest','FontSize',10)
ax = gca;
ax.Box = 'on';
grid on

% %==========================================================================
% Trajectory cost of VI and CVI for different grid sizes
Cost_VI = zeros(2,length(N_set)); % mean and std
Cost_CVI = zeros(2,length(N_set)); % mean and std
Cost_CVI_d = zeros(2,length(N_set)); % mean and std for CVI with dynamic Y_g

for i_N = 1:length(N_set)
    
    N = N_set(i_N);
    str = sprintf('Ex_3_Data_N%2.0f',N);
    load(str)
    temp_ind = ~isinf(Result.tc_VI) & ~isinf(Result.tc_CVI) & ~isinf(Result.tc_CVI_d);
    Cost_VI(1,i_N) = mean(Result.tc_VI(temp_ind));
    Cost_VI(2,i_N) = std(Result.tc_VI(temp_ind));
    Cost_CVI(1,i_N) = mean(Result.tc_CVI(temp_ind));
    Cost_CVI(2,i_N) = std(Result.tc_CVI(temp_ind));
    Cost_CVI_d(1,i_N) = mean(Result.tc_CVI_d(temp_ind));
    Cost_CVI_d(2,i_N) = std(Result.tc_CVI_d(temp_ind));
    
end

h3 = figure;
semilogx(N_vec,Cost_VI(1,:),'Color',[.0 .45 .74],'LineStyle','-','Marker','s','LineWidth',1)
hold on
semilogx(N_vec,Cost_CVI(1,:),'Color',[0 0.5 0],'LineStyle','-','Marker','d','LineWidth',1)
semilogx(N_vec,Cost_CVI_d(1,:),'Color',[.85 .33 .1],'LineStyle','-','Marker','o','LineWidth',1)
hold off
ylabel('Average cost','FontSize',10,'Interpreter','latex')
xlabel('Grid size, $X\times U$','FontSize',10,'Interpreter','latex')
txt1 = 'VI';
txt2 = 'CVI';
txt3 = 'CVI-d';
legend({txt1, txt2, txt3},'Interpreter','latex','Orientation','vertical','Location','northeast','FontSize',10)
ax = gca;
ax.Box = 'on';
grid on


% Saving the results of analysis
str = sprintf('Ex_3_Analysis');
save(str,'Convergence_VI','Convergence_CVI','Convergence_CVI_d',...
    'RunTime_VI','RunTime_CVI','RunTime_CVI_d',...
    'Cost_VI','Cost_CVI','Cost_CVI_d') 