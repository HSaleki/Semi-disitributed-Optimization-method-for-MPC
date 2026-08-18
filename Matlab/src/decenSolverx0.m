function solution = decenSolverx0(data)
%% Problem structure
%       min_x_i sum_i 1/2 x_i' Q x_i + p'_i x_i
% subject to: G_i x_i = g_i
%             H_i x_i <= h_i
% Coupled inequality sum_i A_i x_i <= b

    Q = data.Q;
    p = data.p;
    G = data.G;
    g = data.g;
    H = data.H;
    h = data.h;
    A = data.A;
    b = data.b;
    x0 = data.z0;
    loc_activeset  = data.loc_activeset;
    loc_dual       = data.loc_dual;
    coup_activeset = data.coup_activeset;
    coup_dual      = data.coup_dual;

    N = length(Q);
    dec_var   = cell(1,N);
    dual_l    = cell(1,N);
    dual_c    = zeros(size(b));
    active_l  = cell(1,N);
    active_c  = false(size(b));
    residuals = cell(N+1,1);
    condS     = [];

    % Step 3: Initialize Local Nodes
    nodesx0 = cell(1, N);
    for i = 1:N
        nodesx0{i} = LocalNodex0(i, Q{i}, p{i}, G{i}, g{i}, H{i}, h{i}, A{i}, x0{i}, ...
                                 loc_dual{i}, loc_activeset{i}, coup_dual, coup_activeset);
    end

    % Step 4: Initialize Central Node
    centralx0 = CentralNodex0(nodesx0, b, coup_dual, coup_activeset);

    % Step 5: Optimization Loop
    max_iters = 10000;

    log.change_c = false(1,max_iters);
    log.id       = zeros(1,max_iters);
    log.nA_c     = zeros(1,max_iters);
    log.tau      = zeros(1,max_iters);
    log.is_active_c_hist = cell(1,max_iters);

    for iter = 1:max_iters

        centralx0.aggregateAndSolve();
        exit_flag = centralx0.homotopystep();

        log.change_c(iter) = centralx0.change_c;
        log.id(iter)       = centralx0.id;
        log.nA_c(iter)     = nnz(centralx0.is_active_c);
        log.tau(iter)      = centralx0.tau;
        log.is_active_c_hist{iter} = centralx0.is_active_c;

        if exit_flag == 1
            for i = 1:N
                dec_var{i}   = nodesx0{i}.getdecisionvariable;
                active_l{i}  = nodesx0{i}.getactiveset;
                dual_l{i}    = nodesx0{i}.getdual;
                residuals{i} = nodesx0{i}.getresiduals;
            end
            active_c       = centralx0.getactiveset;
            dual_c         = centralx0.getdual;
            residuals{N+1} = centralx0.getresidual;
            break;
        end
    end

    log.change_c = log.change_c(1:iter);
    log.id       = log.id(1:iter);
    log.nA_c     = log.nA_c(1:iter);
    log.tau      = log.tau(1:iter);
    log.is_active_c_hist = log.is_active_c_hist(1:iter);
    log.n_outer_iters = iter;

    solution.z        = dec_var;
    solution.dual_l   = dual_l;
    solution.dual_c   = dual_c;
    solution.iter     = iter;
    solution.active_l = active_l;
    solution.active_c = active_c;
    solution.cond     = condS;
    solution.res      = residuals;
    solution.log      = log;

end
