classdef CentralNodex0 < handle
    
    %       min_x_i sim_i 1/2 x_i' Q x_i + p'_i x_i
    % subject to: G_i x_i = g_i
    %             H_i x_i <= h_i
    % Coupled inequality sum_i A_i x_i <= b

    properties
        nodes               % Array of LocalNode instances
        nu                  % Global Lagrange multiplier
        delta_nu            % Global Lagrange multiplier change
        % delta_nu_bar       % Coupled dLagrage multiplier
        b                   % Global constraint lower bound vector
        is_active_c         % Active/inactive set for coupled inequality constraint
        n                   % Number of inequality constraints (size(A,1))
        b_tau               % Bound at tau
        b_0                 % Initial bound
        Delta1              % Vector of final gradient and bounds for stopping criterion
        DeltaTau            % Vector of gradient and bounds at tau for stopping criterion
        tau                 % Homotopy parameter
        delta_tau           % Homotopy parameter step
        sigma
        id                  % id of limiting agent that changes active set
        Ax_sum              % Sum_i A_i * x_i   
        Adelta_x_sum        % Sum_i A_i * delta_x_i
        delta_taus          % Vector of delta_tau from all local nodes and central one
        stopflag            % Flag for stopping when reached the accuracy
        flagc               % Flag for reaching coupled constrained accuracy
        exit_flag           % Exit flag for main script
        delta_nu_temp
        S_sum
        rho_sum
        r_b
        change_c
        % change_l
        sigma_tol               % sigma below which the path is closed exactly
        n_zero_c                % consecutive zero-length step counter
        n_zero_budget           % cycle DETECTOR, not a proven bound: with no
                                % anti-cycling rule in place a zero-step cascade
                                % has no a priori length limit, so this only
                                % converts a silent hang into a raised error.
        slack_rel               % warm-start slack relative to row scale;
                                % controls where on the path tight rows are resolved
    end
    
    methods
        function obj = CentralNodex0(nodes, b, coup_dual, coup_activeset)
            obj.nodes = nodes;
            obj.b = b;
            obj.n = length(obj.b);
            obj.nu = coup_dual;                                     % Initialize inequality multipliers
            obj.delta_nu = zeros(obj.n,1);                          % Initialize inequality multipliers change
            % obj.delta_nu_bar = zeros(obj.n,1);
            obj.is_active_c = false(obj.n, 1);
            obj.is_active_c(coup_activeset) = true;                           
            obj.tau = 0;
            obj.delta_tau = 0;
            obj.sigma = 1;
            obj.exit_flag = false;
            obj.delta_taus = zeros(length(obj.nodes) + 1, 1);
            obj.stopflag = zeros(length(obj.nodes), 1);
            obj.flagc = false;
            if any(coup_activeset)
                obj.change_c = true;
            else
                obj.change_c = false;
            end
            obj.Ax_sum = zeros(obj.n,1);
            obj.Adelta_x_sum = zeros(obj.n,1);
            obj.S_sum = zeros(nnz(obj.is_active_c));
            obj.rho_sum = zeros(nnz(obj.is_active_c),1);
            obj.slack_rel = 1e-6;    % = sqrt(activity tol); must precede Initialize
            obj.Initialize();
            obj.sigma_tol = 1e-14;                   % matches the exit test below
            obj.n_zero_c = 0;
            obj.n_zero_budget = 2*( obj.n + ...      % [FIX B17]
                sum(cellfun(@(nd) numel(nd.h), obj.nodes)) );
        end

    function obj = Initialize(obj)
            obj.Delta1 = [obj.b];
            
            if any(obj.is_active_c)
                for i = 1:length(obj.nodes)
                    [Ax_i, Si, rhoi] = obj.nodes{i}.sendinitialvalues();
                    obj.Ax_sum = obj.Ax_sum + Ax_i;
                    obj.S_sum = obj.S_sum + Si;
                    obj.rho_sum = obj.rho_sum + rhoi;
                end
            else
                for i = 1:length(obj.nodes)
                    [Ax_i, ~, ~] = obj.nodes{i}.sendinitialvalues();
                    obj.Ax_sum = obj.Ax_sum + Ax_i;
                end
            end

            % ---- purge weakly active coupled rows ----
            viol  = obj.Ax_sum - obj.b;
            % purge = obj.is_active_c & (obj.nu <= 1e-10*max(norm(obj.nu,inf),1)) ...
            %                        & (viol < 1e-9*max(abs(obj.b),1));
            sc_c  = max(abs(obj.Ax_sum - obj.b), 1);          % or just  D_hard*ones(...)
            nu_s  = max(norm(obj.nu, inf), 1);
            purge = obj.is_active_c & (obj.nu <= 1e-12*nu_s) & (viol < -1e-10*sc_c);

            if any(purge)
                obj.is_active_c(purge) = false;
                obj.nu(purge)          = 0;
            end

            obj.b_0 = obj.b + max(obj.Ax_sum - obj.b, 0);
            obj.b_0(obj.is_active_c) = obj.Ax_sum(obj.is_active_c);
            
            obj.nu(~obj.is_active_c) = 0;
            
            % [FIX B2/B15] slack RELATIVE to the data scale (was an absolute 1e4).
            eps_slack = 1e-10;          % numerical boundary tolerance

            resAx = obj.Ax_sum - obj.b_0;
            too   = ~obj.is_active_c & (resAx > -eps_slack);
            if any(too)
                sc = max([abs(obj.b(too)), abs(obj.Ax_sum(too)), ones(nnz(too),1)], [], 2);
                obj.b_0(too) = obj.Ax_sum(too) + obj.slack_rel*sc;   % strictly interior
            end

            obj.r_b   = obj.b - obj.b_0;
            obj.b_tau = obj.b_0;
            obj.flagc = false;



    end
        
        function obj = updatetauvalues(obj)
            obj.b_tau = obj.b - obj.sigma * obj.r_b;     % == b bitwise at sigma == 0
            % obj.flagc = (obj.sigma == 0);
            obj.flagc = (obj.sigma < 1e-14);
        end


        function obj = aggregateAndSolve(obj)
        
            for pass = 1:obj.n+1
                if ~any(obj.is_active_c)
                    obj.delta_nu(:) = 0;
                    break                               % nodes already stepped in assemblematrices
                end
        
                % ---------- aggregation ----------
                if obj.change_c
                    nA = nnz(obj.is_active_c);
                    obj.S_sum   = zeros(nA);            % sendDataToCentral returns ABSOLUTE S_i
                    obj.rho_sum = zeros(nA,1);
                    for i = 1:length(obj.nodes)
                        [S_i, rho_i, stopflag_i] = obj.nodes{i}.sendDataToCentral();
                        obj.S_sum   = obj.S_sum   + S_i;
                        obj.rho_sum = obj.rho_sum + rho_i;
                        obj.stopflag(i) = stopflag_i;
                        TransactionLogger.getInstance().logData('central', S_i,  'receive');
                        TransactionLogger.getInstance().logData('central', rho_i,'receive');
                    end
                    obj.stopflag(length(obj.nodes)+1) = obj.flagc;
                else
                    [dS, drho, sflag] = obj.nodes{obj.id}.sendchange();
                    obj.S_sum   = obj.S_sum   + dS;
                    obj.rho_sum = obj.rho_sum + drho;
                    obj.stopflag(obj.id) = sflag;
                    TransactionLogger.getInstance().logData('central', dS,  'receive');
                    TransactionLogger.getInstance().logData('central', drho,'receive');
                end
        
                % ---------- rank-revealing test ----------
                [~,R,P] = qr(obj.S_sum,0);
                Rdiag = abs(diag(R));
                tol   = max(size(obj.S_sum)) * eps(Rdiag(1));
                r     = sum(Rdiag > tol);
        
                if r < size(obj.S_sum,2)
                    activeidx = find(obj.is_active_c);
                    obj.dropRows(activeidx(P(r+1:end)));
                    continue                            % re-aggregate on the reduced set
                end
        
                % ---------- solve S dnu = rho - r_b ----------
                obj.delta_nu_temp = obj.S_sum \ ( obj.rho_sum - obj.r_b(obj.is_active_c) );
                obj.delta_nu(:)   = 0;
                obj.delta_nu(obj.is_active_c) = obj.delta_nu_temp;
                obj.broadcastDeltaNu();
                break
            end
        
            obj.aggregateAndSolvestep();                % now unconditional
        end


            
        function dropRows(obj, dropped)
            obj.is_active_c(dropped) = false;
            obj.nu(dropped)          = 0;
        
            for i = 1:length(obj.nodes)
                obj.nodes{i}.dropcoupled(dropped)
            end
        
            obj.change_c = true;
            nA = nnz(obj.is_active_c);
            obj.S_sum   = zeros(nA);          % <-- added
            obj.rho_sum = zeros(nA,1);        % <-- added
        end
        

        function broadcastDeltaNu(obj)
            % Send computed delta_nu back to all local nodes
                for i = 1:length(obj.nodes)
                    obj.nodes{i}.receiveDeltaNu(obj.delta_nu_temp);
                    TransactionLogger.getInstance().logData('central', obj.delta_nu_temp, 'send');
                end
        end




        function obj = aggregateAndSolvestep(obj)
            % Initialize accumulators
            obj.Adelta_x_sum = zeros(obj.n, 1);
            for i = 1:length(obj.nodes)
                [Adelta_x_i, delta_tau_i] = obj.nodes{i}.sendDataToCentralstep();
                obj.Adelta_x_sum = obj.Adelta_x_sum + Adelta_x_i;
                obj.delta_taus(i) = delta_tau_i;
                TransactionLogger.getInstance().logData('central', Adelta_x_i,'receive');
                TransactionLogger.getInstance().logData('central', delta_tau_i,'receive');
            end
                                  
        end


        function exit_flag = homotopystep(obj)



            % --- 1. Ratio Tests ---
            dnu_s  = max(norm(obj.delta_nu(obj.is_active_c), inf), realmin);
            ind_nu = (obj.delta_nu < -1e-10*dnu_s) & obj.is_active_c;

            if ~any(ind_nu)
                delta_tau_da = inf;
                ka = NaN;
            else
                idx     = find(ind_nu);
                ratios  = max( -obj.nu(idx) ./ obj.delta_nu(idx), 0 );   % clamp at 0
                [delta_tau_da, jj] = min(ratios);
                ka = idx(jj);
            end


            candidates = ~obj.is_active_c;
            cand_indx  = find(candidates);
            den   = obj.Adelta_x_sum(candidates) - obj.r_b(candidates);   % = A dx + b_0 - b
            num   = max( obj.b_tau(candidates) - obj.Ax_sum(candidates), 0 );   % clamp at 0
            den_s = max( norm(obj.Adelta_x_sum, inf) + norm(obj.r_b, inf), realmin );

            ind = den > 1e-10*den_s;              % RELATIVE denominator gate

            if ~any(ind)
                delta_tau_p = inf;
                l = NaN;
            else
                temp = num ./ den;
                [delta_tau_p, jj] = min(temp(ind));
                valid_indx = find(ind);
                l = cand_indx(valid_indx(jj));
            end
            
            % Choose smallest valid step
            obj.delta_tau = min([delta_tau_p, delta_tau_da]);
            obj.delta_taus(end) = obj.delta_tau;

            [delta_tau_c, obj.id] = min(obj.delta_taus);
            assert(delta_tau_c >= 0, 'central: negative step %.3e from agent %d', ...
                   delta_tau_c, obj.id);
            obj.delta_tau = delta_tau_c;  
            


            % ---- count zero-length steps BEFORE deciding the step ----
            if min(obj.delta_tau, obj.sigma) <= 0
                obj.n_zero_c = obj.n_zero_c + 1;
            else
                obj.n_zero_c = 0;
            end

            if obj.sigma <= obj.sigma_tol
                actual_step = obj.sigma;
            elseif obj.n_zero_c >= obj.n_zero_budget
                error('CentralNodex0:cycle', ...
                    ['degenerate cycle: %d consecutive zero-length steps at ' ...
                     'sigma = %.6e (sigma_tol = %.3e, budget = %d). ' ...
                     'Remaining perturbation sigma*||r_b|| = %.3e.'], ...
                    obj.n_zero_c, obj.sigma, obj.sigma_tol, obj.n_zero_budget, ...
                    obj.sigma*norm(obj.r_b, inf));
            else
                actual_step = min(obj.delta_tau, obj.sigma);
            end

            % [FIX B17] invariant: the step never exceeds the ratio-test bound.
            assert(actual_step <= obj.delta_tau + 1e-12*max(obj.sigma,1) || ...
                   obj.sigma <= obj.sigma_tol, ...
                   'central: step %.6e exceeds ratio bound %.6e at sigma %.6e', ...
                   actual_step, obj.delta_tau, obj.sigma);


            obj.Ax_sum = obj.Ax_sum + actual_step * obj.Adelta_x_sum;
            obj.nu = obj.nu + actual_step * obj.delta_nu;

            nu_s = norm(obj.nu, inf);
            snap = obj.is_active_c & (abs(obj.nu) <= 4*eps*max(nu_s,1));
            obj.nu(snap) = 0;

            obj.sigma  = obj.sigma - actual_step;            % == 0 bitwise on terminal step
            obj.tau    = 1 - obj.sigma;
            if obj.sigma < 1e-14
                obj.exit_flag = 1;
                for i = 1:length(obj.nodes)
                    obj.nodes{i}.receiveDeltatau(actual_step, NaN, NaN);
                end
                exit_flag = obj.exit_flag;
                return;
            end

            % --- 4. Active Set Logic ---
            % [FIX B9] apply ALL events attaining the minimum, not just the first.
            % tol_tie    = 1e-12 * max(obj.sigma, 1);
            % tied       = obj.delta_taus <= delta_tau_c + tol_tie;
            % local_acts = find(tied(1:end-1));
            tol_tie    = 1e-12 * max(obj.sigma, 1);
            tied       = isfinite(obj.delta_taus) & (obj.delta_taus <= delta_tau_c + tol_tie);
            local_acts = find(tied(1:end-1));

            if tied(end)
                obj.change_c = true;
                if delta_tau_p <= delta_tau_c + tol_tie && ~isnan(l)
                    obj.is_active_c(l) = true;
                end
                if delta_tau_da <= delta_tau_c + tol_tie && ~isnan(ka)
                    obj.is_active_c(ka) = false;
                    obj.nu(ka) = 0;
                end
            else
                % >1 local node acting invalidates the incremental S_sum update
                obj.change_c = numel(local_acts) > 1;
            end

            if obj.change_c
                obj.S_sum   = zeros(nnz(obj.is_active_c));
                obj.rho_sum = zeros(nnz(obj.is_active_c),1);
            end

            for i = 1:length(obj.nodes)
                if any(local_acts == i)
                    obj.nodes{i}.receiveDeltatau(obj.delta_tau, 1, obj.is_active_c);
                else
                    obj.nodes{i}.receiveDeltatau(obj.delta_tau, 0, obj.is_active_c);
                end
            end
            % obj.tau = obj.tau + obj.delta_tau;
            obj.updatetauvalues();
            exit_flag = obj.exit_flag;


        end

        function dual_out = getdual(obj)
            dual_out = obj.nu;
            dual_out(~obj.is_active_c) = 0;
            dual_out(dual_out < 0) = 0;          % clamp; magnitude is O(u)
        end

        function active_out = getactiveset(obj)
            % Export W as held (see LocalNodex0.getactiveset).
            active_out = obj.is_active_c;
        end

        function residuals = getresidual(obj)
            res_ineq = norm( obj.Ax_sum(obj.is_active_c,:) - obj.b(obj.is_active_c,1), inf);
            res_slack = norm( obj.nu' * (obj.b - obj.Ax_sum), inf);
            residuals = [res_ineq; res_slack];
        end

    end
end

function x = back_substitute(A, b)
    % Get the size of the system
    n = length(b);
    
    % Preallocate the solution vector for performance
    x = zeros(n, 1);
    
    % Solve the bottom row first
    x(n) = b(n) / A(n, n);
    
    % Iterate upwards through the matrix
    for i = n-1:-1:1
        % A(i, i+1:n) * x(i+1:n) is the vectorized dot product replacing the inner sum
        x(i) = (b(i) - A(i, i+1:n) * x(i+1:n)) / A(i, i);
    end
end