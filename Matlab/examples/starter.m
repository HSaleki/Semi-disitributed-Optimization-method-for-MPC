clc
clear
close all

% starter.m - closed-loop platoon example for the distributed active-set QP solver.
% Run from anywhere; the line below puts src/ on the path.
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src'));



%% Problem Description
% Simulate and solve MPC (LQR) in this case a platoon of cars   
% Desicion variable defined as z = [u0​;x1​;u1​;x2​;…;uN−1​;xN​] for each agent
%% 1. PARAMETERS (physical units) -------------------------------------
Ts        = 1;                % sampling [s]
nIter     = 50;               % number of simulation iterations (simulation time = Ts*nIter)
N         = 20;               % horizon
nCars     = 5;                % followers
D_hard    = 30;               % safe gap [m] 
D_ref     = 35;               % tracking reference gap
% D_ref should be > D_hard. With D_ref == D_hard the cost minimiser sits
% exactly on the boundary of all nCars*N coupling rows, so every coupling row is
% weakly active (tight with nu = 0) at the solution. That makes the ratio tests
% degenerate  a_i'z − b ​= (p_{i−1,k} ​− p_{i,k}​) −D_hard ​= D_ref ​− D_hard ​= 0
% finds answer but not well behaved

v_ref = 20;               % leader speed to follow [m/s]
u_ref = 0;                % leader jerk to follow 

nx = 3;
nu = 1;

bounds.u_min = -2;   bounds.u_max = 2;         % jerk bounds
bounds.v_min = 0;    bounds.v_max = 30;        % speed bounds
bounds.a_min = -3;   bounds.a_max = 3;         % acceleration bounds

Q = diag([0.3, 0.6, 0.9]);         % state cost matrix
R = 1;                           % input cost matrix

%% 2. STATIC DATA ---------------------------------------------

[A_d,B_d] = chain3rd(Ts);                         % discrete state matrix and input matrix 
[Aeq, Beq] = buildAeqBeqInterleaved(A_d, B_d, N); % Has to be multiplied to x_0 of each car
[K_lqr,P_lqr,~] = dlqr(A_d,B_d,Q,R);              % Terminal cost matrix and optimal state feedback gain
Hessian = buildH_uFirst(Q, R, P_lqr, N);          % All cars are same

%% 3. INITIAL CONDITIONS  -------------------------------------

x_L  = [0; v_ref; 0];          % leader
leader_p = linspace(x_L(1), ...
                         x_L(1)+N*Ts*v_ref, N+1).';

[A_coup, b_coup] = buildCoupling(nCars, nx, nu, N, D_hard, leader_p );
[Aloc, bloc] = buildLocalIneq_uFirst(bounds, N);

x_0 = {[-60;18;-1];
           [-110;25;1];
           [-170;15;0];
           [-240;12;-1];
           [-280;5;1]};

Xs = alloc_nested_cells(nIter+1, nCars, N*nx);
Zs = alloc_nested_cells(nIter+1, nCars, N*(nx+nu));
Xs{1} = x_0;

Xq = alloc_nested_cells(nIter+1, nCars, N*nx);
Xq{1} = x_0;

logs = cell(1,nIter);

iter = zeros(1,nIter);

problem_structure.z0 = Zs{1};
time_decen = zeros(1,nIter);
time_quadprog = zeros(nIter,1);
errors = zeros(1,nIter);

grad = cell(1,nCars);
g = cell(1,nCars);


U_ref = zeros(nu,N);
loc_activeset = repmat({false(length(bloc),1)},1,nCars);
loc_dual = repmat({zeros((length(bloc)+size(Aeq,1)),1)},1,nCars);
coup_activeset = false(length(b_coup),1);
coup_dual = zeros(length(b_coup),1);

Hes = repmat({Hessian},1,nCars);
G = repmat({Aeq},1,nCars);
H = repmat({Aloc},1,nCars);
h = repmat({bloc},1,nCars);

n_eq = size(Aeq,1);
n_ineq = size(Aloc,1);

v0pred = v_ref*ones(nIter+1,1);
a0pred = zeros(nIter+1,1);

zs_quadprog = repmat({cell(1,nCars)},1,nIter);

res = cell(nIter,1);

%% 4. MAIN LOOP --------------------------------------------------------

for time = 1:nIter
    
        if time == 1, d = struct(); end
    assert(isstruct(d), 'diagnostic struct d was overwritten');
    
    TransactionLogger.getInstance().resetCounts();
    [~, b_coup] = buildCoupling(nCars, nx, nu, N, D_hard, leader_p);

    for i = 1:nCars
        % --- build reference for follower i ----------------------------------
        pos_ref  = leader_p(2:end) - i*D_ref;   % N×1   (skip k)
        vel_ref  = 20 * ones(N,1);
        acc_ref  = zeros(N,1);
        
        X_ref_i  = [pos_ref.' ; vel_ref.' ; acc_ref.'];  % 3N×1
        Z_ref_i = buildZref_uFirst(X_ref_i, U_ref);

        grad{i} = - Hessian * Z_ref_i;
        g{i} = Beq * Xs{time}{i};

    end

    problem_structure.Q = Hes;
    problem_structure.p = grad;
    problem_structure.G = G;
    problem_structure.g = g;
    problem_structure.H = H;
    problem_structure.h = h;
    problem_structure.A = A_coup;
    problem_structure.b = b_coup;

    if time >1 
        [problem_structure.z0, sat_, pad_, act_N_, mu_N_, lamN_corr_] = shift_Z(A_d, B_d, Zs{time-1}, leader_p, D_ref, K_lqr, nx, nu, u_ref, ...
            bounds, R, N, P_lqr);        
        
        loc_activeset = shift_active_set(loc_activeset, N);
        loc_dual = shift_dual_eq_ineq(loc_dual, N, nx, n_eq, P_lqr, problem_structure.z0, leader_p, D_ref, v_ref, lamN_corr_);
        
        rowsN_l = (1:6)*N;                           % stage N of each local block
        
        for i = 1:nCars
            loc_activeset{i}(rowsN_l) = false;          % shift_Z owns the stage-N local block
            loc_dual{i}(n_eq + rowsN_l) = 0;
            if act_N_(i) > 0
                loc_activeset{i}(act_N_(i))   = true;
                loc_dual{i}(n_eq + act_N_(i)) = mu_N_(i);
            end
        end
        
        % ----------------------------------------------------------------------
        coup_activeset = shift_active_set(coup_activeset, N);
        coup_dual = shift_dual_ineq(coup_dual, N);

        Ax_ws = zeros(size(b_coup));
        for i = 1:nCars
            Ax_ws = Ax_ws + A_coup{i} * problem_structure.z0{i};
        end
        tol_pred = 10*eps*max(vecnorm(Ax_ws,inf), norm(b_coup,inf));
        rowsN_c = (1:nCars)*N;                       % stage N of each coupling edge
        sc_c = D_hard; 
        coup_activeset(rowsN_c) = (Ax_ws(rowsN_c) - b_coup(rowsN_c)) > tol_pred*sc_c;
        coup_dual(rowsN_c) = 0;
        
    end



    problem_structure.loc_activeset = loc_activeset;
    problem_structure.loc_dual = loc_dual;
    problem_structure.coup_activeset = coup_activeset;
    problem_structure.coup_dual = coup_dual;

    [qp_solution, xquadprog] = quadprog_soln(problem_structure, nx, nu, N);
    zs_quadprog{time} = qp_solution;




    decen_solution = decenSolverx0(problem_structure);
        nCr = numel(problem_structure.Q);
        BQ = blkdiag(problem_structure.Q{:});   Bp = vertcat(problem_structure.p{:});
        BG = blkdiag(problem_structure.G{:});   Bg = vertcat(problem_structure.g{:});
        BH = blkdiag(problem_structure.H{:});   Bh = vertcat(problem_structure.h{:});
        BA = horzcat(problem_structure.A{:});   Bb = problem_structure.b;
        
        opts = optimoptions('quadprog','Display','off', ...
                'OptimalityTolerance',1e-12,'ConstraintTolerance',1e-12,'StepTolerance',1e-14);
        [zstar,~,~,~,lam] = quadprog(BQ,Bp,[BH;BA],[Bh;Bb],BG,Bg,[],[],[],opts);
        
        nl = size(BH,1);
        mu_s = lam.ineqlin(1:nl);      nu_s = lam.ineqlin(nl+1:end);
        rl   = Bh - BH*zstar;          rc   = Bb - BA*zstar;

        sc_p  = D_hard;
        nu_sc = max(1e-12, norm(nu_s, inf));
        mu_sc = max(1e-12, norm(mu_s, inf));
        Wc_star = (rc < 1e-9*sc_p) & (nu_s > 1e-6*nu_sc);
        Wl_star = (rl < 1e-9*sc_p) & (mu_s > 1e-6*mu_sc);
        d.degen(time) = nnz(rc < 1e-9*sc_p) - nnz(Wc_star);

        Wl_warm = vertcat(problem_structure.loc_activeset{:});
        Wc_warm = problem_structure.coup_activeset;
        Wl_end  = vertcat(decen_solution.active_l{:});
        Wc_end  = decen_solution.active_c;
        
        % --- KKT residual of the DECENTRALIZED solution (no reference needed) ---
        zd  = vertcat(decen_solution.z{:});
        lamd = []; mud = [];
        for i = 1:nCr
            lamd = [lamd; decen_solution.dual_l{i}(1:n_eq)];
            mud  = [mud;  decen_solution.dual_l{i}(n_eq+1:end)];
        end
        nud  = decen_solution.dual_c(:);
        ineq = [mud; nud];
        sc_d = max(1, norm(ineq, inf));
        % Wl_star = (rl < 1e-9*sc_p) & (mu_s >  1e-9*sc_d);
        amb_l   = (rl < 1e-9*sc_p) & (mu_s <= 1e-9*sc_d);
        % Wc_star = (rc < 1e-9*sc_p) & (nu_s >  1e-9*sc_d);
        amb_c   = (rc < 1e-9*sc_p) & (nu_s <= 1e-9*sc_d);
        rs   = [BH;BA]*zd - [Bh;Bb];
        d.rho(time) = max([ ...
            norm(BQ*zd + Bp + BG'*lamd + BH'*mud + BA'*nud, inf)/max(1,norm(Bp,inf)); ...
            norm(BG*zd - Bg, inf)/max(1,norm(Bg,inf)); ...
            max(0,  max(rs))/sc_p; ...
            max(0, -min(ineq))/sc_d; ...
            norm(ineq.*rs, inf)/(sc_p*sc_d) ]);
        
        d.warm_c(time) = nnz(xor(Wc_warm,Wc_star));  d.warm_l(time) = nnz(xor(Wl_warm,Wl_star));
        d.end_c(time)  = nnz(xor(Wc_end, Wc_star));  d.end_l(time)  = nnz(xor(Wl_end, Wl_star));
        d.amb(time)    = nnz(amb_c) + nnz(amb_l);
        d.err(time)    = norm(zd - zstar, inf);
        d.iter(time)   = decen_solution.iter;
        d.kmin(time)   = d.warm_c(time) + d.warm_l(time);
        

    Zs{time} = decen_solution.z;
    iter(time) = decen_solution.iter;
    loc_activeset = decen_solution.active_l;
    loc_dual = decen_solution.dual_l;
    coup_activeset = decen_solution.active_c;
    coup_dual = decen_solution.dual_c;
    
    res{time} = decen_solution.res;
      
 
    loc_act{time} = loc_activeset;
    coup_ac{time} = coup_activeset;


    logs{time} = decen_solution.log;


    % Retrieve the logger counts
    logger = TransactionLogger.getInstance();
    counts = logger.getCounts();
    
    % Save the central counts (for example, send/receive counts from central node)
    centralCount{time} = counts.central;
    
    % For localCount, we want a cell array where each element corresponds to one local node.
    % counts.localBreakdown is a containers.Map whose keys are node ids.
    localBreakdownKeys = counts.localBreakdown.keys;         % Cell array of node id keys (as strings)
    localBreakdownValues = counts.localBreakdown.values;       % Cell array of corresponding count structures
    % Optionally, you can sort these keys if needed.
    localCount{time} = localBreakdownValues;  % Now each cell element (at time t) is a cell array 
                                              % with one element per local node.
    centralCounttotal{time} = centralCount{time}.send.doubleCount + centralCount{time}.receive.doubleCount;
    for num = 1: nCars
        localCounttotal{time}{num} = localCount{time}{num}.send.doubleCount + localCount{time}{num}.receive.doubleCount;
    end
    % Reset the logger for the next QP iteration:
    logger.resetCounts();

    Xs{time+1} = simulate(A_d,B_d,Xs{time},Zs{time});
    Xq{time+1} = simulate(A_d,B_d,Xq{time},zs_quadprog{time});

    leader_p = [ leader_p(2:end); leader_p(end) + Ts* 20 ];


        error_loc = 0;
    for i=1:nCars
        error_loc=error_loc+max(abs(Zs{time}{i}-zs_quadprog{time}{i}));
    end
    errors(time) = error_loc;
        
end

quad_time = mean(time_quadprog);
decen_time = mean(time_decen);

%% 5. PLOTs

    %    A_d, B_d  
    %    x_0{i}           
    %    leader trajectory, v0pred, a0pred, etc. same as before
    
    %—————————————————————————————
    %  1) PREALLOCATE over simulation time
    %—————————————————————————————
    pos = zeros(nCars, nIter+1);
    vel = zeros(nCars, nIter+1);
    acc = zeros(nCars, nIter+1);
    
    % fill in the "time = 0" column from x0_phys
    for i = 1:nCars
        pos(i,1) = x_0{i}(1);
        vel(i,1) = x_0{i}(2);
        acc(i,1) = x_0{i}(3);
    end
    
    %—————————————————————————————
    %  2) CLOSED‐LOOP SIMULATION: reconstruct x[k+1] from U*[1]
    %—————————————————————————————
    for t = 1:nIter
        for i = 1:nCars
            % store into the "time = t+1" column
            xkp1 = Xs{t+1}{i};
            pos(i,t+1) = xkp1(1);
            vel(i,t+1) = xkp1(2);
            acc(i,t+1) = xkp1(3);
        end
    end
    
    
    tt = (0:nIter)*Ts;
    cFollowers = lines(nCars);
    cLeader    = [0 0 0];
    
    labelsF = "Car " + (1:nCars);
    labelL  = "Leader";
    
    % 1) positions
    figure; hold on
    plot(tt, linspace( x_L(1), x_L(1)+nIter*Ts*v_ref, nIter+1 )', ...
         'Color',cLeader,'LineWidth',2);
    for i = 1:nCars
        plot(tt, pos(i,:), 'Color',cFollowers(i,:),'LineWidth',1.6);
    end
    xlabel('time [s]'); ylabel('position p [m]');
    title('Predicted positions'); grid on
    legend([labelL, labelsF],'Location','best')
    
    % 2) velocities
    figure; hold on
    plot(tt, v0pred, 'Color',cLeader,'LineWidth',2)
    for i = 1:nCars
        plot(tt, vel(i,:), 'Color',cFollowers(i,:),'LineWidth',1.6);
    end
    xlabel('time [s]'); ylabel('velocity v [m/s]');
    title('Predicted velocities'); grid on
    legend([labelL, labelsF],'Location','best')
    
    % 3) accelerations
    figure; hold on
    plot(tt, a0pred, 'Color',cLeader,'LineWidth',2)
    for i = 1:nCars
        plot(tt, acc(i,:), 'Color',cFollowers(i,:),'LineWidth',1.6);
    end
    xlabel('time [s]'); ylabel('acceleration a [m/s²]');
    title('Predicted accelerations'); grid on
    legend([labelL, labelsF],'Location','best')
    
    % 4) gaps to leader
    % leader's position at each *physical* time:
    leader_pos_sim = linspace( x_L(1), x_L(1)+nIter*Ts*v_ref, nIter+1 );
    figure; hold on
    for i = 1:nCars
        plot(tt, leader_pos_sim - pos(i,:), 'Color',cFollowers(i,:),'LineWidth',1.6);
    end
    if exist('D','var')
        yline(D,'--k','D','LineWidth',1);
    end
    xlabel('time [s]'); ylabel('leader gap [m]');
    title('Gap to leader vs. time'); grid on
    legend("Car "+(1:nCars),'Location','best')


  



%% 5b. TOTAL DATA TRANSFERRED PER TIME STEP -------------------------------
%   C(t) = cs_d(t) + cr_d(t)                central node, tx + rx
%   L(t) = sum_i [ ls_d(i,t) + lr_d(i,t) ]  all local nodes, tx + rx

    tK   = (1:nIter)*Ts;
    Ctot = cell2mat(centralCounttotal);                      % 1 x nIter
    Ltot = cellfun(@(c) sum(cell2mat(c)), localCounttotal);  % 1 x nIter

    figure; hold on; grid on
    plot(tK, Ctot, 'LineWidth',1.6);
    plot(tK, Ltot, 'LineWidth',1.6);
    xlabel('time [s]'); ylabel('doubles per step');
    title('Data transferred per closed-loop step');
    legend({'central (tx+rx)','local (tx+rx, all nodes)'},'Location','best');
%% 5c. RUN SUMMARY ---------------------------------------------------------
%  Everything below is already collected in the loop; nothing is recomputed.

    K = max(iter,1);

    fprintf('\n=== closed loop: %d steps, N = %d, nCars = %d ===\n', nIter, N, nCars);

    % (1) correctness of the decentralised solution, independent of quadprog:
    %     rho = max relative violation of stationarity / feasibility /
    %           dual feasibility / complementarity
    fprintf('KKT residual rho        max %.3e   median %.3e\n', max(d.rho), median(d.rho));

    % (2) agreement with the reference solver
    fprintf('||z_d - z*||_inf        max %.3e   median %.3e\n', max(d.err), median(d.err));

    % (3) active-set identification at exit (symmetric difference vs W*)
    fprintf('|W_end XOR W*|          coupled %d   local %d   (steps not identified: %d)\n', ...
            sum(d.end_c), sum(d.end_l), nnz(d.end_c + d.end_l));

    % (4) degeneracy / ambiguity: rows tight with mu = 0, where W* is not unique
    fprintf('weakly active rows      coupled %d   ambiguous (l+c) %d\n', ...
            sum(d.degen), sum(d.amb));

    % (5) iterations vs the warm-start lower bound k_min = |W_warm XOR W*|
    fprintf('outer iterations        total %d   mean %.2f   max %d\n', ...
            sum(iter), mean(iter), max(iter));
    fprintf('warm-start distance k   mean %.2f   max %d   steps with k = 0: %d\n', ...
            mean(d.kmin), max(d.kmin), nnz(d.kmin == 0));
    fprintf('excess iters (K - k)    mean %.2f   max %d   steps with K = 1: %d\n', ...
            mean(iter - d.kmin), max(iter - d.kmin), nnz(iter == 1));

    % (6) communication
    fprintf('data per step [doubles] central %.0f   local %.0f   (total %.2f MB)\n', ...
            mean(Ctot), mean(Ltot), 8*(sum(Ctot)+sum(Ltot))/1e6);
    fprintf('data per iteration      central %.0f   local %.0f\n', ...
            sum(Ctot)/sum(K), sum(Ltot)/sum(K));

    % (7) worst step, for drilling in
    [~, tw] = max(d.rho);
    fprintf('worst step t = %d: rho %.3e, K = %d, k_min = %d, |A_c| = %d\n', ...
            tw, d.rho(tw), iter(tw), d.kmin(tw), nnz(coup_ac{tw}));
%% 6. HELPER FUNCTIONS

function [qp_solution, xquadprog] = quadprog_soln(problem_structure, nx, nu, N)
    nCars = length(problem_structure.Q);
    BigQ = blkdiag(problem_structure.Q{:});
    Bigp = vertcat(problem_structure.p{:});
    BigG = blkdiag(problem_structure.G{:});
    Bigg = vertcat(problem_structure.g{:});
    BigH = blkdiag(problem_structure.H{:});
    Bigh = vertcat(problem_structure.h{:});
    A_coupl = horzcat(problem_structure.A{:});

    BigQ  = 0.5*(BigQ + BigQ');
    BiGHA = [BigH; A_coupl];
    Bigh  = [Bigh; problem_structure.b];

    options = optimoptions(@quadprog, 'Display','off', ...
        'OptimalityTolerance',1e-12, 'ConstraintTolerance',1e-12, ...
        'StepTolerance',1e-14, 'MaxIterations',2000);

 
    [xquadprog, ~, ef, out] = quadprog(BigQ, Bigp, BiGHA, Bigh, BigG, Bigg, [], [], [], options);

    if ef ~= 1
        warning('quadprog_soln:notConverged', ...
            'exitflag=%d after %d iters (%s)', ef, out.iterations, out.message);
    end

    qp_solution = cell(1,nCars);
    for i = 1:nCars
        qp_solution{i} = xquadprog((i-1)*(nx+nu)*N+1 : i*(nx+nu)*N);
    end
end


function [Z_shifted, sat, pad, act_N, mu_N, lamN_corr] = shift_Z(A, B, Z_cell, p_leader, D, k_lqr, nx, nu, u_ref, bounds, R, N, P)
    v_ref_tail = 20;
    Cv = [0 1 0];   Ca = [0 0 1];
    Reff = R + B.'*P*B;

    n = numel(Z_cell);
    Z_shifted = cell(1,n);
    sat = zeros(1,n);  pad = zeros(1,n);
    act_N = zeros(1,n);  mu_N = zeros(1,n);  lamN_corr = zeros(nx,n);

    cc   = [ 1; -1;  Cv*B; -(Cv*B);  Ca*B; -(Ca*B) ];    % coeff of u
    rows = (1:6).'*N;                                    % row index in Aloc
    gx   = [zeros(nx,2), Cv.', -Cv.', Ca.', -Ca.'];      % d(row)/d x_N

    for i = 1:n
        Z  = Z_cell{i};
        Zs = zeros(size(Z));
        Zs(1:end-(nx+nu)) = Z(nx+nu+1:end);

        xNm1  = Z(end-nx+1:end);                          % old x_N = new x_{N-1}
        xrNm1 = [p_leader(end-1) - i*D; v_ref_tail; 0];
        u_lqr = u_ref - k_lqr*(xNm1 - xrNm1);
        pad(i) = u_lqr;

        xfree = A*xNm1;
        dd = [ bounds.u_max; -bounds.u_min; ...
               bounds.v_max - Cv*xfree; -bounds.v_min + Cv*xfree; ...
               bounds.a_max - Ca*xfree; -bounds.a_min + Ca*xfree ];

        iu = find(cc > 0);   il = find(cc < 0);
        [hi, ku] = min(dd(iu)./cc(iu));   ku = iu(ku);
        [lo, kl] = max(dd(il)./cc(il));   kl = il(kl);

        kr = 0;

        if lo > hi                                        % terminal stage infeasible
            u_pad = min(max(u_lqr, bounds.u_min), bounds.u_max);
            warning('shift_Z:tailInfeasible','car %d: lo=%.3g > hi=%.3g', i, lo, hi);
        elseif u_lqr > hi
            u_pad = hi;   kr = ku;
        elseif u_lqr < lo
            u_pad = lo;   kr = kl;
        else
            u_pad = u_lqr;  kr = 0;
        end

        if kr > 0
            act_N(i) = rows(kr);
            mu_N(i)  = 2*Reff*(u_lqr - u_pad)/cc(kr);
            lamN_corr(:,i) = -gx(:,kr)*mu_N(i);
        end

        sat(i) = max(u_lqr - hi, lo - u_lqr);             % >0 iff a stage-N row binds
        Zs(end-(nx+nu)+1:end) = [u_pad; A*xNm1 + B*u_pad];
        Z_shifted{i} = Zs;
    end
end


function [A,B] = chain3rd(Ts)
    A = [1    Ts  0.5*Ts^2;
         0    1         Ts;
         0    0          1];
    B = [Ts^3/6; Ts^2/2; Ts];
end

function [Aeq, Beq] = buildAeqBeqInterleaved(A_d, B_d, N)
% buildAeqBeqInterleaved  Build dense equality constraint matrices for MPC
%
% Dynamics:
%   x_{k+1} = A_d x_k + B_d u_k,  k=0..N-1
%
% Interleaved u-first decision variable:
%   z = [u0; x1; u1; x2; ...; u_{N-1}; xN]
%
% Equality constraints in the form:
%   Aeq * z = Beq * x0
%
% Inputs:
%   A_d : (nx x nx)
%   B_d : (nx x nu)
%   N   : horizon length (positive integer)
%
% Outputs:
%   Aeq : (N*nx) x (N*(nu+nx))   (same for all cars)
%   Beq : (N*nx) x nx            (same for all cars)

    % --- sizes ---
    nx = size(A_d, 1);
    nu = size(B_d, 2);

    % --- checks ---
    if size(A_d,2) ~= nx
        error('A_d must be square (nx x nx).');
    end
    if size(B_d,1) ~= nx
        error('B_d must have nx rows.');
    end
    if ~isscalar(N) || N < 1 || floor(N) ~= N
        error('N must be a positive integer.');
    end

    % --- dimensions ---
    nZ  = N*(nu + nx);   % length of z
    nEq = N*nx;          % number of equality equations

    Aeq = zeros(nEq, nZ);
    Beq = zeros(nEq, nx);

    % block indices in z = [u0;x1;u1;x2;...;u_{N-1};xN]
    idx_u = @(k) ( (k-1)*(nu+nx) + (1:nu) );       % u_{k-1} stored in pair k
    idx_x = @(k) ( (k-1)*(nu+nx) + nu + (1:nx) );  % x_k stored in pair k

    % Build constraints:
    % For k=1..N:
    %   x_k - A_d x_{k-1} - B_d u_{k-1} = 0
    for k = 1:N
        rows = (k-1)*nx + (1:nx);

        % + x_k
        Aeq(rows, idx_x(k)) = eye(nx);

        % - B_d u_{k-1}
        Aeq(rows, idx_u(k)) = -B_d;

        if k >= 2
            % - A_d x_{k-1}
            Aeq(rows, idx_x(k-1)) = -A_d;
        else
            % k=1: RHS = A_d*x0
            Beq(rows, :) = A_d;
        end
    end
end

function Ep = P_selector_uFirst(nx, nu, N)
% buildEp_uFirst  Build Ep such that P = Ep*z
%
% z = [u0; x1; u1; x2; ...; u_{N-1}; xN]
% P = [p(1); p(2); ...; p(N)]
%
% Assumes position is the FIRST component of the state x = [p; ...].

    Ep = zeros(N, N*(nu+nx));
    Cp = [1, zeros(1, nx-1)];   % selects position from x

    for k = 1:N
        idx_xk = (k-1)*(nu+nx) + nu + (1:nx);  % columns for x_k inside z
        Ep(k, idx_xk) = Cp;
    end
end

function [Acell, b] = buildCoupling(M, nx, nu, N, D, P0)

% - Followers are cars 1..M-1, each has decision variable z_i (interleaved u-first):
%       z_i = [u0; x1; u1; x2; ...; u_{N-1}; xN]
%
% Safety constraints (gap >= D):
%   for i = 1..M1:   P_i - P_{i-1} <= -D*1
%
% Since P_0 is known, the first constraint becomes:
%   P_1 <= P_0 - D*1
%
% We build distributed matrices such that:
%   sum_{i=1}^{M} A_i z_i <= b
%
% Inputs:
%   M  : number of cars 
%   nx : state dimension
%   nu : input dimension
%   N  : horizon length
%   D  : safety distance (e.g. 30)
%   P0 : (N x 1) vector of leader predicted positions [p0(1)..p0(N)]
%
% Outputs:
%   Acell : cell array Acell{1..M}, each (M*N) x (N*(nu+nx))
%   b     : (M*N) x 1

    % if M < 2
    %     error('M must be >= 2 (leader + at least one follower).');
    % end
    % if numel(P0) ~= N
    %     error('P0 must be N x 1 (predicted leader positions over horizon).');
    % end
    P0 = P0(2:end,:);

    % Build Ep so that P_i = Ep*z_i
    Ep = P_selector_uFirst(nx, nu, N);

    % Sizes
    nFollowers = M;
    nCoupled   = nFollowers * N;            % total # coupling inequalities
    nZi        = N*(nu + nx);               % size of local decision z_i

    % Allocate Acell
    Acell = cell(nFollowers, 1);
    for i = 1:nFollowers
        Acell{i} = zeros(nCoupled, nZi);
    end

    % Global RHS: default is -D for all edges and all steps
    b = -D * ones(nCoupled, 1);

    % --- Edge 1: between leader (0) and follower (1)
    % Constraint: P1 - P0 <= -D  =>  P1 <= P0 - D
    rows1 = 1:N;
    Acell{1}(rows1, :) = Ep;               % +Ep*z1
    b(rows1) = P0 - D*ones(N,1);           % move P0 to RHS

    % --- Edges e = 2..(M): between follower (e-1) and follower (e)
    % Constraint: P_e - P_{e-1} <= -D
    % => +Ep*z_e  -Ep*z_{e-1} <= -D
    for e = 2:nFollowers
        rows = (e-1)*N + (1:N);

        % +Ep on car e
        Acell{e}(rows, :) = Acell{e}(rows, :) + Ep;

        % -Ep on car (e-1)
        Acell{e-1}(rows, :) = Acell{e-1}(rows, :) - Ep;

        % RHS already has -D for this block
    end
end

function [Aloc, bloc] = buildLocalIneq_uFirst(bounds, N)
% Local inequality constraints for one car (dense)
% z = [u0; x1; u1; x2; ...; u_{N-1}; xN]
% x = [p; v; a], u = jerk
%
% Constraints:
%   u_min <= u(k) <= u_max, k=0..N-1
%   v_min <= v(k) <= v_max, k=1..N
%   a_min <= a(k) <= a_max, k=1..N

    % --- dimensions (fixed for platoon model) ---
    nx = 3;   % [p; v; a]
    nu = 1;   % jerk

    % --- bounds ---
    u_min = bounds.u_min;   u_max = bounds.u_max;
    v_min = bounds.v_min;   v_max = bounds.v_max;
    a_min = bounds.a_min;   a_max = bounds.a_max;

    % --- z size ---
    nZ = N*(nu + nx);

    % --- Build selectors Ev, Ea, Eu ---
    Eu = zeros(N, nZ);   % picks input    u(k-1) from u-block
    Ev = zeros(N, nZ);   % picks velocity v(k) from x_k
    Ea = zeros(N, nZ);   % picks accel    a(k) from x_k

    % selectors inside x=[p;v;a]
    Cv = [0 1 0];   % picks v
    Ca = [0 0 1];   % picks a

    for k = 1:N
        % indices of u_{k-1} and x_k inside interleaved z
        idx_u = (k-1)*(nu+nx) + (1:nu);          % u_{k-1}
        idx_x = (k-1)*(nu+nx) + nu + (1:nx);     % x_k

        Eu(k, idx_u) = 1;        % u_{k-1}
        Ev(k, idx_x) = Cv;       % v_k
        Ea(k, idx_x) = Ca;       % a_k
    end

    % --- Stack inequalities: Ev, Ea, Eu ---
    Aloc = [ Eu;
            -Eu;
             Ev;
            -Ev;
             Ea;
            -Ea];

    bloc = [ u_max*ones(N,1);
            -u_min*ones(N,1);
             v_max*ones(N,1);
            -v_min*ones(N,1);
             a_max*ones(N,1);
            -a_min*ones(N,1)];
end


function H = buildH_uFirst(Q, R, P, N)
% H for z = [u0;x1;u1;x2;...;u_{N-1};xN]
% Cost: sum u'R u + sum x'Q x + terminal x'P x
% Output: J = 0.5*z'*H*z + f'*z + const

    nx = size(Q,1);
    nu = size(R,1);

    nZ = N*(nu+nx);
    H  = zeros(nZ,nZ);

    idx_u = @(k) ( (k-1)*(nu+nx) + (1:nu) );
    idx_x = @(k) ( (k-1)*(nu+nx) + nu + (1:nx) );

    for k = 1:N
        Qk = Q;
        if k == N
            Qk = P;
        end

        H(idx_u(k), idx_u(k)) = 2*R;
        H(idx_x(k), idx_x(k)) = 2*Qk;
    end
end

function zref = buildZref_uFirst(xref, uref)
% zref for z = [u0;x1;u1;x2;...;u_{N-1};xN]
% xref is (nx x N):  xref(:,k) = x_k_ref
% uref is (nu x N):  uref(:,k) = u_{k-1}_ref

    nx = size(xref,1);
    N  = size(xref,2);
    nu = size(uref,1);

    zref = zeros(N*(nu+nx),1);

    idx_u = @(k) ( (k-1)*(nu+nx) + (1:nu) );
    idx_x = @(k) ( (k-1)*(nu+nx) + nu + (1:nx) );

    for k = 1:N
        zref(idx_u(k)) = uref(:,k);
        zref(idx_x(k)) = xref(:,k);
    end
end

function Z = alloc_nested_cells(N, n, k)
    Z = cell(1, N);
    for i = 1:N
        Zi = cell(1, n);
        for j = 1:n
            Zi{j} = zeros(k, 1);
        end
        Z{i} = Zi;
    end
end

function x_next = simulate(A,B,x,Z)
    num_sys = length(Z);
    x_next = cell(num_sys,1);
    for i=1:num_sys
        x_next{i} = A*x{i}+B*Z{i}(1,:);
    end
end

function activeset = shift_active_set(active_set, N)
    if ~iscell(active_set)
        % 1. Reshape into a 20x5 matrix
        M = reshape(active_set, N, []); 
        M_new = false(size(M)); % Initialize new matrix as all false
        
        M_new(1:N-1, :) = M(2:N, :);
        M_new(N, :) = M(N, :);
        
        activeset = reshape(M_new, [], 1);
    else
        n = length(active_set);
        activeset = cell(n,1);
        for i = 1:n
            M = reshape(active_set{i}, N, []);
            M_new = false(size(M));
        
            M_new(1:N-1, :) = M(2:N, :);
            M_new(N, :) = M_new(N, :) | M(N, :);
        
            activeset{i} = reshape(M_new, [], 1);
        end
    end
end


function dual_new = shift_dual_eq_ineq(dual_, N_, nx, n_eq_, P_, z0, leader_p, D_ref, v_ref, lamN_corr)
    % Lambda dual for equality
    % Mu dual for inequality
    n = length(dual_);
    dual_new = cell(n,1);
    for i = 1:n
        xRefN = [leader_p(end) - i*D_ref ; v_ref ; 0];
        x_N = z0{i}(end-nx+1:end,1);

    lambda_old = dual_{i}(1:n_eq_);
    lambda_mat = reshape(lambda_old, nx, N_);

    lambda_mat_new = zeros(nx, N_);
    lambda_mat_new(:, 1:N_-1) = lambda_mat(:, 2:end);

    lambda_mat_new(:, N_) = -2 * P_ * (x_N - xRefN) + lamN_corr(:,i);

    lambda_new = lambda_mat_new(:);

    mu_old = dual_{i}(n_eq_+1:end);
    mu_mat = reshape(mu_old, N_, []);

    mu_new = zeros(size(mu_mat));
    mu_new(1:N_-1, :) = mu_mat(2:N_, :);
    mu_new(N_, :) = mu_mat(N_, :);
    
    mu_new = reshape(mu_new, [], 1);

    dual_new{i} = [lambda_new; mu_new];
    end
end


function dual_new = shift_dual_ineq(dual_, N)
        % 1. Reshape into a 20x5 matrix
        M = reshape(dual_, N, []); 
        M_new = zeros(size(M)); % Initialize new matrix as all false
        
        M_new(1:N-1, :) = M(2:N, :);
        % [FIX B5] must mirror shift_active_set: set and dual share one convention.
        M_new(N, :) = M(N, :);
        
        dual_new = reshape(M_new, [], 1);
end


function s = ternary(c,a,b) 
    if c
        s=a; 
    else
        s=b; 
    end 
end
