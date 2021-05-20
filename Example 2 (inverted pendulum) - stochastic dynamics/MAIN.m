% This file is an example of implementation of the CVI and VI algorithms 
% for solving the infinite-horizon optimal control problem for a noisy
% inverted pendulum. 
% Please see Section 4.2 of the the following article for more details:
%   
%   M.A.S. Kolarijani, C.F. Max, and P. Mohajerin Esfahani (2020), 
%   Fast Approximate Dynamic Programming for Infinite-Horizon 
%   Continuous-State Markov Decision Processes, 
%   preprint arXiv:2102.08880.
% 

clc
clear all

%% Optimal control problem instances (initial states and horizon) 
T = 400; % horizon
NumInstances = 50; % the number of instances (initial states)

% Initial states
initial_state_set = cell(NumInstances,1); % the set of initial states (allocation)
theta_lim = pi/4; thetadot_lim = pi; u_lim = 3; % state and input variables limits
Delta_x = [-theta_lim,theta_lim; -thetadot_lim,thetadot_lim]; % i-th row determines the range of i-th state
for i_Inst = 1:NumInstances
    initial_state_set{i_Inst} = Delta_x(:,1) + (Delta_x(:,2) - Delta_x(:,1)).*rand(size(Delta_x,1),1);
end

% Stochasticity 
W1 = theta_lim*[-.05, -.025, 0, .025, .05];
W2 = thetadot_lim*[-.05, -.025, 0, .025, .05];
W = combvec(W1,W2); % discrete set of disturbance (column vectors)
pmf_W = ones(1,size(W,2))/size(W,2); % disturbance probability mass function

disturb_seq = cell(NumInstances,1);
for i_Inst = 1:NumInstances
    ind_w_t = randsample(length(pmf_W),T,true,pmf_W);
    disturb_seq{i_Inst} = W(:,ind_w_t);
end

%% Implementation of algorithms and solving the problem instances

% Size of the grids for discretization
N_set = [11 15 21 25 31 35 41]; % grid sizes 11*11, ...

i_N = 1;
save GS_IS_Data

while i_N <= length(N_set) % iteraton over grid sizes
    
    clear all
    
    % Optimal control problem instances (initial states and horizon)
    load GS_IS_Data
    
    %======================================================================
    
    % (1) Problem data (inputs given by the user)

    % Dynamics: f(x,u) = f_s(x) + B*u 
    %    parameters -------------------------------------------------------
    J1 = 1.91e-4; m = .055; g = 9.81; el = .042; b = 3.0e-6; K = 53.6e-3; R = 9.50; tau = 0.05;
    alpha = m*g*el/J1; beta = -(b+K^2/R)/J1; gamma = K/(J1*R);
     %---------------------------------------------------------------------
    state_dynamics = @(x) ( x + tau * [x(2); (alpha*sin(x(1)) + beta*x(2)) ]);  
    input_matrix = tau*[0;gamma]; 
    
    % Stochasticity 
    Stoch = true;

    % Constraints
    theta_lim = pi/4; thetadot_lim = pi; u_lim = 3; % state and input variables limits
    x_constraints = @(x) ([1 0; -1 0; 0 1; 0 -1]*x - [theta_lim; theta_lim; thetadot_lim; thetadot_lim]); %  the box [-pi/4,pi/4] * [-pi,pi]
    u_constraints = @(u) ([1;-1]*u - u_lim*ones(2,1)); % interval [-3,3]

    % Cost functions
    state_cost = @(x) (x'*x); % quadratic stage cost (state)
    input_cost = @(u) (u'*u); % quadratic stage cost (input)

    % Conjugate of input-dependent stage cost
    Delta_u = [-u_lim,u_lim]; % i-th row determines the range of i-th input
    conj_input_cost = @(v) (conj_Quad_box(1,Delta_u(:,1),Delta_u(:,2),v));
    
    % Discouont factor
    gamma = 0.95;

    % Discretization of the state and input (and their dual) spaces    
    %   State space discretization (uniform grid)
    N = N_set(i_N)*[1,1];
    N_x = N; % vector of number of data points in the discretization of state space in EACH dimension
    Delta_x = [-theta_lim,theta_lim; -thetadot_lim,thetadot_lim]; % i-th row determines the range of i-th state
    state_grid = unif_grid(Delta_x,N_x); % so the discretization of the state space is a uniform grid of size 11^2 over [-1,1]^2 
    %   Input space discretization (uniform grid) - similar to state space discretization described above
    N_u = N; 
    Delta_u = [-u_lim,u_lim]; 
    input_grid = unif_grid(Delta_u,N_u); 
    %   The state dual grid (Y_g):
    N_y = N_x; 
    %   The grid (Z_g):
    N_z = N_x;
    %   The input dual grid (V_g):
    N_v = N_u;
    %   Set the value of the coefficeint "alpha" 
    alpha_y = 1;

    VI_implementation = true;

    if VI_implementation && Stoch
        fast_stoch_DP = false;
    end

    if VI_implementation 
        VI_interpol_mehtod = 'linear';
        VI_extrapol_mehtod = 'linear';
    end

    if Stoch 
        CVI_interpol_mehtod = 'linear';
        CVI_extrapol_mehtod = 'linear';
    end

    % Termination criterion
    epsilon = 0.001;
    %======================================================================
    
    % (2) Problem data (data objects defined for internal use)

    ProblemData = struct;

    % Dynamics
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
    ProblemData.ConjugateInputCost = conj_input_cost; 

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

    % Configuration of VI algorithms
    if VI_implementation
        VI_Data = struct;
        if Stoch
            VI_Data.FastStochastic = fast_stoch_DP ;
        end
        VI_Data.ExtensionInterpolationMethod = VI_interpol_mehtod;
        VI_Data.ExtensionExtrapolationMethod = VI_extrapol_mehtod;
        VI_Data.TerminationError = epsilon;
    end

    CVI_Data = struct;
    CVI_Data.DualGridConstructionAlpha = alpha_y;
    CVI_Data.TerminationError = epsilon;
    if Stoch
        CVI_Data.ExpectationInterpolationMethod = CVI_interpol_mehtod;
        CVI_Data.ExpectationExtrapolationMethod = CVI_extrapol_mehtod;
    end
    %======================================================================
    
    % Feasibility check of discretization scheme
    feasibility_check(ProblemData)
    
    %======================================================================
    
    % (3) Implementation and Results
    
    %  Output data
    Result = struct;
    Result.InitialState = initial_state_set;
    if Stoch
        Result.DisturbanceSequence = disturb_seq;
    end
    Result.rt_CVI_nd = 0;
    Result.tc_CVI_nd= zeros(NumInstances,1);
    Result.rt_CVI_ns = 0;
    Result.tc_CVI_ns= zeros(NumInstances,1);
    Result.rt_CVI_ad = 0;
    Result.tc_CVI_ad= zeros(NumInstances,1);
    Result.rt_VI = 0;
    Result.tc_VI= zeros(NumInstances,1);
    %----------------------------------------------------------------------
    
    % CVI Algorithm ("N"umerical conjugation and "D"ynamic Y_g)
    
    CVI_Data.NumericalConjugateCost = true;
    CVI_Data.DynamicDualGridConstruction = true;
    
    [temp1, temp2, temp3] = CVI_Alg(ProblemData, CVI_Data);
    Result.V_CVI_nd = temp1;
    Result.conv_CVI_nd = temp2;
    Result.rt_CVI_nd = sum(temp3);

    for i_Inst = 1:NumInstances % Iteraton over control problem instances

        ProblemData.InitialState = Result.InitialState{i_Inst}; 
        if Stoch
            ProblemData.DisturbanceSequence = Result.DisturbanceSequence{i_Inst}; 
        end

        [dummy1, dummy2, temp4] = forward_iter(ProblemData, temp1);
        Result.tc_CVI_nd(i_Inst) = temp4;
    end
    
    %----------------------------------------------------------------------
    
    % CVI Algorithm ("N"umerical conjugation and "S"tatic Y_g)
    
    CVI_Data.NumericalConjugateCost = true;
    CVI_Data.DynamicDualGridConstruction = false;
    
    [temp5, temp6, temp7] = CVI_Alg(ProblemData, CVI_Data);
    Result.V_CVI_ns = temp5;
    Result.conv_CVI_ns = temp6;
    Result.rt_CVI_ns = sum(temp7);

    for i_Inst = 1:NumInstances % Iteraton over control problem instances

        ProblemData.InitialState = Result.InitialState{i_Inst}; 
        if Stoch
            ProblemData.DisturbanceSequence = Result.DisturbanceSequence{i_Inst}; 
        end

        [dummy3, dummy4, temp8] = forward_iter(ProblemData, temp5);
        Result.tc_CVI_ns(i_Inst) = temp8;
    end

    %----------------------------------------------------------------------
    
    % CVI Algorithm ("A"nalytical conjugation and "D"ynamic Y_g)
    
    CVI_Data.NumericalConjugateCost = false;
    CVI_Data.DynamicDualGridConstruction = true;
     
    [temp9, temp10, temp11] = CVI_Alg(ProblemData, CVI_Data);
    Result.V_CVI_ad = temp9;
    Result.conv_CVI_ad = temp10;
    Result.rt_CVI_ad = sum(temp11);

    for i_Inst = 1:NumInstances % Iteraton over control problem instances

        ProblemData.InitialState = Result.InitialState{i_Inst}; 
        if Stoch
            ProblemData.DisturbanceSequence = Result.DisturbanceSequence{i_Inst}; 
        end

        [dummy5, dummy6, temp12] = forward_iter(ProblemData, temp9);
        Result.tc_CVI_ad(i_Inst) = temp12;
    end

    %----------------------------------------------------------------------
    
    % VI Algorithm
    if VI_implementation

        [temp13, temp14, temp15] = VI_Alg(ProblemData, VI_Data);
        Result.V_VI = temp13;
        Result.conv_VI = temp14;
        Result.rt_VI = temp15;

        for i_Inst = 1:NumInstances % Iteraton over control problem instances

            ProblemData.InitialState = Result.InitialState{i_Inst};
            if Stoch
                ProblemData.DisturbanceSequence = Result.DisturbanceSequence{i_Inst}; 
            end

            [dummy7, dummy8, temp16] = forward_iter(ProblemData, temp13);
            Result.tc_VI(i_Inst) = temp16;

        end

    end
    
    %----------------------------------------------------------------------

    % Saving the results for each grid size 
    str = sprintf('Ex_2s_Data_N%2.0f',N(1));
    save(str,'ProblemData','VI_Data','CVI_Data','Result')
    
    % Next grid size
    i_N = i_N+1;
    save('GS_IS_Data','i_N','-append')

end

%% Analysis of results

% Convergence rate of VI and CVI for largest grid size
i_N = length(N_set);
N = N_set(i_N);
str = sprintf('Ex_2s_Data_N%2.0f',N);
load(str)

Convergence_VI = Result.conv_VI(2:end);
Convergence_CVI_nd = Result.conv_CVI_nd(2:end); % "N"umerical conjugation and "D"ynamic Y_g
Convergence_CVI_ns = Result.conv_CVI_ns(2:end); % "N"umerical conjugation and "S"tatic Y_g
Convergence_CVI_ad = Result.conv_CVI_ad(2:end); % "A"nalytical conjugation and "D"ynamic Y_g

gamma_rate = zeros(1,length(Convergence_VI));
for i=1:length(Convergence_VI)
    gamma_rate(i) = 0.95^(i-1)* Convergence_VI(1);
end

h1 = figure;
semilogy(1:length(Convergence_VI),Convergence_VI,'Color',[.0 .45 .74],'LineStyle','-','Marker','none','LineWidth',1.5)
hold on
semilogy(1:length(Convergence_CVI_ns),Convergence_CVI_ns,'Color',[.93 .69 .13],'LineStyle',':','Marker','none','LineWidth',1.5)
semilogy(1:length(Convergence_CVI_nd),Convergence_CVI_nd,'Color',[.85 .33 .1],'LineStyle','--','Marker','none','LineWidth',1.5)
semilogy(1:length(gamma_rate),gamma_rate,'Color','k','LineStyle','-.','Marker','none','LineWidth',1.5)
hold off
ylabel('$\Vert V - V_+ \Vert_{\infty}$','FontSize',10,'Interpreter','latex')
xlabel('Iteration, $k$','FontSize',10,'Interpreter','latex')
txt1 = 'VI';
txt2 = 'CVI-s';
txt3 = 'CVI-d';
legend({txt1, txt2, txt3},'Interpreter','latex','Orientation','vertical','Location','northeast','FontSize',10)
ax = gca;
ax.Box = 'on';
grid on

%==========================================================================
% Running time of VI and CVI for different grid sizes
RunTime_VI = zeros(1,length(N_set));
RunTime_CVI_nd = zeros(1,length(N_set)); % "N"umerical conjugation and "D"ynamic Y_g
RunTime_CVI_ns = zeros(1,length(N_set)); % "N"umerical conjugation and "S"tatic Y_g
RunTime_CVI_ad = zeros(1,length(N_set)); % "A"nalytical conjugation and "D"ynamic Y_g

for i_N = 1:length(N_set)
    
    N = N_set(i_N);
    str = sprintf('Ex_2s_Data_N%2.0f',N);
    load(str)
    
    RunTime_VI(i_N) = Result.rt_VI;
    RunTime_CVI_nd(i_N) = Result.rt_CVI_nd;
    RunTime_CVI_ns(i_N) = Result.rt_CVI_ns;
    RunTime_CVI_ad(i_N) = Result.rt_CVI_ad;
    
end

h2 = figure;
N_vec = (N_set.*N_set)/100;
loglog(N_vec,RunTime_VI,'Color',[.0 .45 .74],'LineStyle','-','Marker','s','LineWidth',1)
hold on
loglog(N_vec,RunTime_CVI_ns,'Color',[.93 .69 .13],'LineStyle','-','Marker','d','LineWidth',1)
loglog(N_vec,RunTime_CVI_nd,'Color',[.85 .33 .1],'LineStyle','-','Marker','o','LineWidth',1)
hold off
ylabel('Time (sec)','FontSize',10,'Interpreter','latex')
xlabel('Grid size, $N$ $(\times 100)$','FontSize',10,'Interpreter','latex')
txt1 = 'VI';
txt2 = 'CVI-s';
txt3 = 'CVI-d';
legend({txt1, txt2, txt3},'Interpreter','latex','Orientation','vertical','Location','northwest','FontSize',10)
ax = gca;
ax.Box = 'on';
grid on

%==========================================================================
% Trajectory cost of VI and CVI for different grid sizes
Cost_VI = zeros(2,length(N_set)); % mean and std
Cost_CVI_nd = zeros(2,length(N_set)); % mean and std for "N"umerical conjugation and "D"ynamic Y_g
Cost_CVI_ns = zeros(2,length(N_set)); % mean and std for "N"umerical conjugation and "S"tatic Y_g
Cost_CVI_ad = zeros(2,length(N_set)); % mean and std for "A"nalytical conjugation and "D"ynamic Y_g

for i_N = 1:length(N_set)
    
    N = N_set(i_N);
    str = sprintf('Ex_2s_Data_N%2.0f',N);
    load(str)
    temp_ind = ~isinf(Result.tc_VI) & ~isinf(Result.tc_CVI_nd) & ~isinf(Result.tc_CVI_ns);
    Cost_VI(1,i_N) = mean(Result.tc_VI(temp_ind));
    Cost_VI(2,i_N) = std(Result.tc_VI(temp_ind));
    Cost_CVI_nd(1,i_N) = mean(Result.tc_CVI_nd(temp_ind));
    Cost_CVI_nd(2,i_N) = std(Result.tc_CVI_nd(temp_ind));
    Cost_CVI_ns(1,i_N) = mean(Result.tc_CVI_ns(temp_ind));
    Cost_CVI_ns(2,i_N) = std(Result.tc_CVI_ns(temp_ind));
    Cost_CVI_ad(1,i_N) = mean(Result.tc_CVI_ad(temp_ind));
    Cost_CVI_ad(2,i_N) = std(Result.tc_CVI_ad(temp_ind));
    
end

h3 = figure;
plot(N_vec,Cost_VI(1,:),'Color',[.0 .45 .74],'LineStyle','-','Marker','s','LineWidth',1)
hold on
plot(N_vec,Cost_CVI_ns(1,:),'Color',[.93 .69 .13],'LineStyle','-','Marker','d','LineWidth',1)
plot(N_vec,Cost_CVI_nd(1,:),'Color',[.85 .33 .1],'LineStyle','-','Marker','o','LineWidth',1)
hold off
ylabel('Cost','FontSize',10,'Interpreter','latex')
xlabel('Grid size, $N$ $(\times 100)$','FontSize',10,'Interpreter','latex')
txt1 = 'VI';
txt2 = 'CVI-s';
txt3 = 'CVI-d';
legend({txt1, txt2, txt3},'Interpreter','latex','Orientation','vertical','Location','northeast','FontSize',10)
ax = gca;
ax.Box = 'on';
grid on

%==========================================================================
% Difference between outpus of VI and CVI for different grid sizes
VF_diff_nd = zeros(2,length(N_set)); % mean and std for "N"umerical conjugation and "D"ynamic Y_g
VF_diff_ns = zeros(2,length(N_set)); % mean and std for "N"umerical conjugation and "S"tatic Y_g
VF_diff_ad = zeros(2,length(N_set)); % mean and std for "A"nalytical conjugation and "D"ynamic Y_g

for i_N = 1:length(N_set)
    
    N = N_set(i_N);
    str = sprintf('Ex_2s_Data_N%2.0f',N);
    load(str)
    
    temp1 = abs(Result.V_CVI_nd(:) - Result.V_VI(:));
    VF_diff_nd(1,i_N) = mean(temp1(~isinf(temp1)));
    VF_diff_nd(2,i_N) = std(temp1(~isinf(temp1)));
    
    temp2 = abs(Result.V_CVI_ns(:) - Result.V_VI(:));
    VF_diff_ns(1,i_N) = mean(temp2(~isinf(temp2)));
    VF_diff_ns(2,i_N) = std(temp2(~isinf(temp2)));
    
    temp3 = abs(Result.V_CVI_ad(:) - Result.V_VI(:));
    VF_diff_ad(1,i_N) = mean(temp3(~isinf(temp3)));
    VF_diff_ad(2,i_N) = std(temp3(~isinf(temp3)));
    
end

h4 = figure;
x = N_vec; alpha = 0.3;
y = VF_diff_ns(1,:); z = VF_diff_ns(2,:);
plot(N_vec,VF_diff_ns(1,:),'Color',[.93 .69 .13],'LineStyle','-','Marker','d','LineWidth',1)
hold on
plot(N_vec,VF_diff_nd(1,:),'Color',[.85 .33 .1],'LineStyle','-','Marker','o','LineWidth',1)
fill([x fliplr(x)],[y+z fliplr(y-z)],[.93 .69 .13],'FaceAlpha',alpha,'linestyle','none');
y = VF_diff_nd(1,:); z = VF_diff_nd(2,:);
fill([x fliplr(x)],[y+z fliplr(y-z)],[.85 .33 .1],'FaceAlpha',alpha,'linestyle','none');
hold off
ylabel('Mean absolute difference','FontSize',10,'Interpreter','latex')
xlabel('Grid size, $N$ $(\times 100)$','FontSize',10,'Interpreter','latex')
txt3 = 'CVI-s';
txt4 = 'CVI-d';
legend({txt3, txt4},'Interpreter','latex','Orientation','vertical','Location','northeast','FontSize',10)
ax = gca;
ax.Box = 'on';
grid on

% Saving the results of analysis
str = sprintf('Ex_2s_Analysis');
save(str,'Convergence_VI','Convergence_CVI_nd','Convergence_CVI_ns', 'Convergence_CVI_ad',...
    'RunTime_VI','RunTime_CVI_nd','RunTime_CVI_ns','RunTime_CVI_ad',...
    'Cost_VI','Cost_CVI_nd','Cost_CVI_ns','Cost_CVI_ad',...
    'VF_diff_nd','VF_diff_ns','VF_diff_ad')
