classdef LocalNodex0 < handle
    %       min_x 1/2 x' B x + p' x
    % subject to: G x = g
    %             H x <= h
    

    %nodesx0{i} = LocalNodex0(i, Q{i}, p{i}, G{i}, g{i}, H{i}, h{i}, A{i}, x0{i}, loc_dual{i}, loc_activeset{i}, coup_dual, coup_activeset);
    properties %(Access = private)
        id                      % Unique identifier
        B                       % Quadratic term in local objective function
        p                       % Linear term
        G                       % Local equality constraint matrix
        g                       % Local equality constraint vector
        H                       % Local inequality constraint matrix
        h                       % Local inequality constraint vector
        A                       % Local matrix of the global coupled constraint
        x                       % Local decision variable
        delta_x                 % Local decision variable step
        delta_x_bar
        nx                      % Dimension of decision variable
        neq_l                   % Number of local equality constraints
        nineq_l                 % Number of local inequality constraints
        nineq_c                 % Number of coupld inequality constraints
        lambda                  % Lagrange multipliers for equality constraints
        delta_lambda            % Lagrange multipliers change for equality constraints
        mu                      % Lagrange multipliers for inequality constraints
        delta_mu                % Lagrange multipliers change for inequality constraints
        nu                      % Lagrange multipliers for coupled inequality constraints
        delta_nu                % Lagrange multipliers change for coupled inequality constraints
        %delta_nu_bar
        is_active_l             % Active/inactive set for local inequality constraint
        is_active_c             % Active/inactive set for coupled inequality constraint 
        tau                     % Homotopy parameter   
        delta_tau               % Change in tau in each iteration
        sigma                   % 1 - tau
        d_p
        d_g
        d_h
        % r                       % Perturbation matrix
        h_tau                   % Inequality bound at tau
        p_tau                   % Gradeitn vector at tau
        g_tau                   % Equality constraint at tau    
        h_0                     % Initial inequality bound
        p_0                     % Initial gradient vector
        g_0                     % Initial equality constraint vector
        Delta1                  % Vector of final gradient and bounds for stopping criterion
        DeltaTau                % Vector of gradient and bounds at tau for stopping criterion
        flag                    % Stopping criterion reached flag = 1
        delta_tau_p             % delta_tau from primal ratio
        delta_tau_d             % delta_tau from dual ratio
        k_p                     % Index of inactive constraint becoming active
        k_d                     % Index of active constraint with sign change
        idx                     % Temporary index for finding blocking index
        nA_l                    % Number of active local inequalities
        nA_c                    % Number of active coupled inequalities
        change_l                % Change in local active set
        change_c                % Change in coupled actives set     
        GHA                     % Active constraint matrix
        r_                      % Rank of active constraint matrix
        V                       % Matrix storing the essential Householder vectors $v_k$
        beta                    % Vector of Householder multipliers
        Qi
        Yi
        Zi
        Ti
        YBYi
        ZBZi
        YBZi
        ZBYi
        s_Yi
        s_Zi
        s_Zi_bar
        Li
        Di
        r_pi
        r_gi
        r_ghi
        ZBYs_Yi
        YBYs_Yi
        YBZs_Zi
        Yr_pi
        r_sZi
        r_sZi_bar
        dxbar_i
        ZAT_i
        YAT_i
        Mi
        S_i
        rho_i
        delta_S_i
        delta_rho_i
        Zg                      % [DEP] null(G), cached  (nx x m)
        Hz                      % [DEP] H*Zg, reduced inequality rows (nineq_l x m)
        GGt                     % [DEP] G*G', for exact lambda compensation on a drop
        slack_rel               % warm-start slack, RELATIVE to row scale.
                                %   h_0,k = H_k x + slack_rel*max(1,|h_k|,|H_k x|)
                                % Sets WHERE on the path a warm-start-tight row is
                                % resolved:  t*_k = s_k/(s_k + H_k dx).
                                %   slack_rel >> |H dx|/scale  ->  t* -> 1 (late)
                                %   slack_rel << |H dx|/scale  ->  t* -> 0 (early)
                                % Late is better on long/cold paths: the row is
                                % picked up near the optimum instead of being
                                % activated at a bad point and dropped again.

    end
    
    %LocalNodex0(i, Q{i}, p{i}, G{i}, g{i}, H{i}, h{i}, A{i}, x0{i}, loc_activeset{i}, loc_dual{i}, coup_dual, coup_activeset);
    methods
        
        function obj = LocalNodex0(id, B, p, G, g, H, h, A, x0, loc_dual, loc_activeset, coup_dual, coup_activeset)
            obj.id = id;
            obj.B = B;
            obj.nx = size(B,1);
            %obj.B = obj.B + 1e-8*diag(Q);        % Small regulaization for numerical stability
            obj.p = p;
            obj.G = G;
            obj.g = g;
            obj.neq_l = size(obj.G,1);
            obj.H = H;
            obj.nineq_l = size(obj.H,1);
            obj.h = h;
            obj.A = A;
            obj.nineq_c = size(A,1);
            if isempty(obj.G)
                obj.Zg  = eye(obj.nx);
                obj.GGt = [];
            else
                obj.Zg  = null(full(obj.G));
                obj.GGt = obj.G * obj.G.';
            end
            obj.Hz  = obj.H * obj.Zg;
            obj.x = x0;
            obj.lambda = loc_dual(1:length(obj.g));
            obj.delta_lambda = zeros(length(obj.g),1);
            obj.mu = loc_dual(length(obj.g)+1:end);          
            obj.delta_mu = zeros(obj.nineq_l,1);
            obj.is_active_l = false(obj.nineq_l,1);
            obj.is_active_l(loc_activeset) = true;
            obj.mu(~obj.is_active_l) = 0;
            obj.nu = coup_dual;
            obj.delta_nu = zeros(size(A,1),1);
            obj.is_active_c = false(obj.nineq_c,1);
            obj.is_active_c(coup_activeset) = true;
            obj.nu(~obj.is_active_c) = 0;
            obj.flag = false;
            obj.tau = 0;
            obj.delta_tau = 0;
            obj.sigma = 1;
            obj.p_0 = zeros(obj.nx,1);
            obj.g_0 = zeros(length(obj.g),1);
            obj.h_0 = zeros(obj.nineq_l,1);
            obj.change_l = false;
            obj.change_c = false;
            obj.slack_rel = 1e-6;   % = sqrt(activity tol): smallest slack that
                                    % keeps a near-boundary row strictly inactive
                                    % at sigma = 1 without inflating ||d_h||.
            % obj.initial_lineardependence();
            obj.Initialize();
            % obj.updatetauvalues();
            % obj.assembleKKT();
        end


        function obj = Initialize(obj)

            obj.g_0 = obj.G * obj.x;
            Hx = obj.H * obj.x;


            


            obj.h_0 = obj.h + max(Hx - obj.h, 0);
            obj.h_0(obj.is_active_l) = Hx(obj.is_active_l);

            obj.mu(~obj.is_active_l) = 0;

            eps_slack = 1e-10;

            resHx = Hx - obj.h_0;
            too   = ~obj.is_active_l & (resHx > -eps_slack);
            if any(too)
                sc = max([abs(obj.h(too)), abs(Hx(too)), ones(nnz(too),1)], [], 2);
                obj.h_0(too) = Hx(too) + obj.slack_rel*sc;
            end
          
            obj.nA_l = nnz(obj.is_active_l);                                % # of local active constraints
            obj.nA_c = nnz(obj.is_active_c);                                % # of coupled active constraints
            
            obj.GHA = [obj.G', obj.H(obj.is_active_l,:)'];                  % Active constraint set [G' ; H_A']
            
            [obj.V, obj.beta, T, keptIdx, obj.r_] = seqQR_rankreveal(obj.GHA);           % Hauseholder qr with Householder vectors V, Householder multipliers beta ...
                                                                            % R upper triangulat part of R in qr, keptIdx index of columns kept, r rank of original matrix
            obj.Ti = T;

            if any(~keptIdx)

                [obj.is_active_l, obj.mu, obj.lambda, ~, ind_dropped] = ...
                    obj.resolveDependency(obj.is_active_l, obj.mu, obj.lambda);


                obj.nA_l = nnz(obj.is_active_l);
                obj.GHA  = [obj.G', obj.H(obj.is_active_l,:)'];
                [obj.V, obj.beta, obj.Ti, keptIdx2, obj.r_] = seqQR_rankreveal(obj.GHA);
                assert(all(keptIdx2), ...
                    'node %d: still rank deficient after ratio-test drop', obj.id);

                sc = max([abs(obj.h(ind_dropped)), abs(Hx(ind_dropped)), ...
                          ones(numel(ind_dropped),1)], [], 2);
                obj.h_0(ind_dropped) = Hx(ind_dropped) + obj.slack_rel*sc;
            end


            % Compute adapted gradient p
            if ~isempty(obj.G)
                obj.p_0 = - (obj.B*obj.x + obj.G'*obj.lambda + obj.H'*obj.mu + obj.A'*obj.nu);
            else
                obj.p_0 = - (obj.B*obj.x + obj.H'*obj.mu + obj.A'*obj.nu);
            end

            obj.d_p = obj.p_0 - obj.p;
            obj.d_g = obj.g_0 - obj.g;
            obj.d_h = obj.h_0 - obj.h;
            
            
            obj.p_tau = obj.p_0;
            obj.g_tau = obj.g_0;
            obj.h_tau = obj.h_0;

            obj.flag = false;

            obj.r_pi = obj.d_p;
            obj.r_gi = - obj.d_g;

            % [B    GH'] [ dx  ] = [ r_p  ]
            % [GH    0 ] [ dlm ] = [ r_gh ]


            obj.Qi = eye(obj.nx);
            for j = obj.r_:-1:1
                v = obj.V(j:obj.nx, j);
                obj.Qi(j:obj.nx,:) = obj.Qi(j:obj.nx,:) - obj.beta(j) * v * (v.'*obj.Qi(j:obj.nx,:));
            end 
            
            obj.Yi = obj.Qi(:,1:obj.r_);
            obj.Zi = obj.Qi(:,obj.r_+1:end);

            obj.YBYi = obj.Yi' * obj.B * obj.Yi;   % Y' * B * Y
            obj.ZBYi = obj.Zi' * obj.B * obj.Yi;   % Z' * B * Y
            obj.ZBZi = obj.Zi' * obj.B * obj.Zi; % Z' * B * Z

            
            obj.Yr_pi = obj.Yi' * obj.r_pi;
            Zr_pi = obj.Zi' * obj.r_pi;


            if any(obj.is_active_l)
                obj.r_ghi = [obj.r_gi; - obj.d_h(obj.is_active_l)];
            else
                obj.r_ghi = [obj.r_gi];
            end
            
            obj.s_Yi = obj.Ti' \ obj.r_ghi;

            obj.YBYs_Yi = obj.YBYi * obj.s_Yi;

            [obj.Li, obj.Di] = ldl(obj.ZBZi);
            
            obj.ZBYs_Yi = obj.ZBYi * obj.s_Yi;            
            
            obj.r_sZi_bar  = Zr_pi  - obj.ZBYs_Yi; 

            t1 = obj.Li \ obj.r_sZi_bar;
            t2 = obj.Di \ t1;
            obj.s_Zi_bar = obj.Li' \ t2;

           
            obj.dxbar_i = obj.Qi * [obj.s_Yi; obj.s_Zi_bar];

            if any(obj.is_active_c)
                obj.ZAT_i = obj.Zi' * obj.A(obj.is_active_c,:)';
                % obj.ZAT_i = obj.QA(obj.r_+1:end,:); 

                t1 = obj.Li \ obj.ZAT_i;
                t2 = obj.Di \t1;
                obj.Mi = obj.Li' \ t2;
    
                obj.S_i = obj.ZAT_i' * obj.Mi;

                obj.rho_i = obj.A(obj.is_active_c,:) * obj.dxbar_i;
                % need comunicate and get \delta nu
            else
                obj.s_Zi = obj.s_Zi_bar;
                
                % obj.Yr_pi = obj.QTr(1:obj.r_);

                obj.delta_x = obj.dxbar_i;
                obj.YBZs_Zi = obj.ZBYi'* obj.s_Zi;
                rhs_lm = obj.Yr_pi - obj.YBYs_Yi - obj.YBZs_Zi;
                delta_lm = obj.Ti \ rhs_lm;
                obj.delta_lambda = delta_lm(1:obj.neq_l);
                delta_mu_temp = delta_lm(obj.neq_l+1:end);
                obj.delta_mu(:) = 0;
                obj.delta_mu(obj.is_active_l) = delta_mu_temp;

                obj.homotopystep();
            end


        end

        function [Ax, Si, rhoi] = sendinitialvalues(obj)
            Ax = obj.A * obj.x;
            Si = obj.S_i;
            rhoi = obj.rho_i;
            if any(obj.is_active_c)
                TransactionLogger.getInstance().logData('local', Ax, 'send', obj.id);
                TransactionLogger.getInstance().logData('local', Si, 'send', obj.id);
                TransactionLogger.getInstance().logData('local', rhoi, 'send', obj.id);
            else
                TransactionLogger.getInstance().logData('local', Ax, 'send', obj.id);
            end
        end


        function obj = updatetauvalues(obj)
            % Affine in sigma. At sigma == 0 this returns p, g, h bitwise.
            obj.p_tau = obj.p + obj.sigma * obj.d_p;
            obj.g_tau = obj.g + obj.sigma * obj.d_g;
            obj.h_tau = obj.h + obj.sigma * obj.d_h;
        
            % obj.flag = (obj.sigma == 0);
            obj.flag = (obj.sigma < 1e-14);

            obj.assemblematrices();
        end


        function obj = assemblematrices(obj)

            obj.nA_l = nnz(obj.is_active_l);                                % # of local active constraints
            obj.nA_c = nnz(obj.is_active_c);                                % # of coupled active constraints

            if obj.change_l
                % [B    GH'] [ dx  ] = [ r_p  ]
                % [GH    0 ] [ dlm ] = [ r_gh ]
                             
                obj.GHA = [obj.G', obj.H(obj.is_active_l,:)'];                  % Active constraint set [G' ; H_A']
                
                [obj.V, obj.beta, T, keptIdx, obj.r_] = seqQR_rankreveal(obj.GHA);     % Hauseholder qr with Householder vectors V, Householder multipliers beta ...
                                                                            % R upper triangulat part of R in qr, keptIdx index of columns kept, r rank of original matrix
                obj.Ti = T;

                if obj.r_ < min(size(obj.GHA))
                    ind_among_act = find(~keptIdx) - obj.neq_l;
                    if any(ind_among_act <= 0)
                        error('node %d: equality block G is rank deficient (r_=%d, neq_l=%d)', ...
                              obj.id, obj.r_, obj.neq_l);
                    end
                    % [FIX DEP] same ratio-test drop as in Initialize: choose the
                    % leaving row so that H'mu + G'lambda is unchanged.

                    [obj.is_active_l, obj.mu, obj.lambda, ~, ind_dropped] = ...
                        obj.resolveDependency(obj.is_active_l, obj.mu, obj.lambda);


                    obj.delta_mu(ind_dropped) = 0;

                    obj.nA_l = nnz(obj.is_active_l);
                    obj.GHA  = [obj.G', obj.H(obj.is_active_l,:)'];
                    [obj.V, obj.beta, T, ~, obj.r_] = seqQR_rankreveal(obj.GHA);
                    obj.Ti = T;
                    if obj.r_ < min(size(obj.GHA))
                        error('node %d: still rank deficient after drop (r_=%d, cols=%d)', ...
                              obj.id, obj.r_, size(obj.GHA,2));
                    end
                end

                obj.Qi = eye(obj.nx);
                for j = obj.r_:-1:1
                    v = obj.V(j:obj.nx, j);
                    obj.Qi(j:obj.nx,:) = obj.Qi(j:obj.nx,:) - obj.beta(j) * v * (v.'*obj.Qi(j:obj.nx,:));
                end 
                
                obj.Yi = obj.Qi(:,1:obj.r_);
                obj.Zi = obj.Qi(:,obj.r_+1:end);
    
                obj.YBYi = obj.Yi' * obj.B * obj.Yi;   % Y' * B * Y
                obj.ZBYi = obj.Zi' * obj.B * obj.Yi;   % Z' * B * Y
                obj.ZBZi = obj.Zi' * obj.B * obj.Zi; % Z' * B * Z
    
                
                obj.Yr_pi = obj.Yi' * obj.r_pi;
                Zr_pi = obj.Zi' * obj.r_pi;
 
                if any(obj.is_active_l)
                    obj.r_ghi = [obj.r_gi; -obj.d_h(obj.is_active_l)];
                else
                    obj.r_ghi = [obj.r_gi];
                end
                
                obj.s_Yi = obj.Ti' \ obj.r_ghi;
                            
                obj.YBYs_Yi = obj.YBYi * obj.s_Yi;
    
                [obj.Li, obj.Di] = ldl(obj.ZBZi);
    
                obj.ZBYs_Yi = obj.ZBYi * obj.s_Yi;               
                obj.r_sZi_bar  = Zr_pi  - obj.ZBYs_Yi;

                t1 = obj.Li \ obj.r_sZi_bar;
                t2 = obj.Di \ t1;
                obj.s_Zi_bar = obj.Li' \ t2;
                
                obj.dxbar_i = obj.Qi * [obj.s_Yi; obj.s_Zi_bar];
            end

            if any(obj.is_active_c)
                if obj.change_c || obj.change_l

                    obj.ZAT_i = obj.Zi' * obj.A(obj.is_active_c,:)';
                end
                % obj.ZAT_i = obj.QA(obj.r_+1:end,:);

                t1 = obj.Li \ obj.ZAT_i;
                t2 = obj.Di \t1;
                obj.Mi = obj.Li' \ t2;

                if obj.change_c
                    obj.S_i = obj.ZAT_i' * obj.Mi;
                    obj.rho_i  = obj.A(obj.is_active_c,:) * obj.dxbar_i;
                elseif obj.change_l
                    S_i_temp = obj.ZAT_i' * obj.Mi;
                    rho_i_temp = obj.A(obj.is_active_c,:) * obj.dxbar_i;
                    obj.delta_S_i = S_i_temp - obj.S_i;
                    obj.delta_rho_i = rho_i_temp - obj.rho_i;

                    obj.S_i = S_i_temp;
                    obj.rho_i = rho_i_temp;
                end
            else
                obj.s_Zi = obj.s_Zi_bar;

                obj.dxbar_i = obj.Qi * [obj.s_Yi; obj.s_Zi_bar];

                obj.delta_x = obj.dxbar_i;
                obj.YBZs_Zi  = obj.ZBYi' * obj.s_Zi_bar;
                rhs_lm = obj.Yr_pi - obj.YBYs_Yi - obj.YBZs_Zi;
                delta_lm = obj.Ti \ rhs_lm;
                obj.delta_lambda = delta_lm(1:obj.neq_l);
                delta_mu_temp = delta_lm(obj.neq_l+1:end);
                obj.delta_mu(:) = 0;
                obj.delta_mu(obj.is_active_l) = delta_mu_temp;

                obj.homotopystep();
            end
        end

      
        function [S_i, rho_i_bar, flag] = sendDataToCentral(obj)
            % Return local matrix and vector to be sent to central node
            S_i = obj.S_i;
            rho_i_bar = obj.rho_i;
            TransactionLogger.getInstance().logData('local', obj.S_i, 'send', obj.id);
            TransactionLogger.getInstance().logData('local', obj.rho_i, 'send', obj.id);
            flag = obj.flag;
        end


        function receiveDeltaNu(obj, delta_nu_temp)
            % Store received delta_nu from central node
            obj.delta_nu(:) = 0;
            obj.delta_nu(obj.is_active_c) = delta_nu_temp;
            TransactionLogger.getInstance().logData('local', delta_nu_temp, 'receive', obj.id);
  
            obj.s_Zi = obj.s_Zi_bar - obj.Mi * obj.delta_nu(obj.is_active_c); 


            obj.delta_x = obj.Qi * [obj.s_Yi; obj.s_Zi];

            obj.YBZs_Zi = obj.ZBYi' * obj.s_Zi;

            % obj.YAT_i = obj.QA(1:obj.r_,:); 
            obj.YAT_i = obj.Yi' * obj.A(obj.is_active_c,:)';
            
            rhs_lm = obj.Yr_pi - obj.YBYs_Yi - obj.YBZs_Zi - obj.YAT_i * obj.delta_nu(obj.is_active_c);  
            delta_lm = obj.Ti \ rhs_lm;
            obj.delta_lambda = delta_lm(1:obj.neq_l);
            delta_mu_temp = delta_lm(obj.neq_l+1:end);
            obj.delta_mu(:) = 0;
            obj.delta_mu(obj.is_active_l) = delta_mu_temp;



            obj.homotopystep();            
        end
        

        function obj = homotopystep(obj)
        
            % ---------- dual ratio test ----------
            mu_s  = max(norm(obj.mu(obj.is_active_l), inf), 1);
            dmu_s = max(norm(obj.delta_mu(obj.is_active_l), inf), realmin);
        
            snap = obj.is_active_l & (abs(obj.mu) <= 4*eps*mu_s);
            obj.mu(snap) = 0;                                   

            % Descending multipliers only, gated RELATIVE to ||delta_mu||_inf.
            neg = (obj.delta_mu < -1e-10*dmu_s) & obj.is_active_l;

            if ~any(neg)
                obj.delta_tau_d = inf;   obj.k_d = NaN;
            else
                neg_idx = find(neg);
                ratios  = max( -obj.mu(neg) ./ obj.delta_mu(neg), 0 );   % clamp at 0
                [obj.delta_tau_d, jj] = min(ratios);
                obj.k_d = neg_idx(jj);
            end

            % ---------- primal ratio test ----------
            cand          = ~obj.is_active_l;
            inactive_indx = find(cand);
            Hdx   = obj.H(cand,:) * obj.delta_x;
            num   = max( obj.h_tau(cand,:) - obj.H(cand,:) * obj.x, 0 );   % clamp at 0
            den   = Hdx + obj.d_h(cand,:);
            den_s = max([norm(Hdx, inf) + norm(obj.d_h, inf), norm(obj.h, inf), 1]);

            ind = den > 1e-10*den_s;              % RELATIVE denominator gate
            if ~any(ind)
                obj.delta_tau_p = inf;   obj.k_p = NaN;
            else
                temp = num ./ den;
                [obj.delta_tau_p, jj] = min(temp(ind));
                valid_indx = find(ind);
                obj.k_p = inactive_indx(valid_indx(jj));
            end
        
            obj.delta_tau = min(obj.delta_tau_p, obj.delta_tau_d);
            assert(isscalar(obj.delta_tau) && obj.delta_tau >= 0, ...
                   'node %d: bad delta_tau', obj.id);
        end


        function [Adeltaxi, delta_taui] = sendDataToCentralstep(obj)
            % Return local matrix and vector to be sent to central node
            Adeltaxi = obj.A * obj.delta_x;
            delta_taui = obj.delta_tau;
            TransactionLogger.getInstance().logData('local', Adeltaxi, 'send', obj.id);
            TransactionLogger.getInstance().logData('local', delta_taui, 'send', obj.id);
        end
        

        function [dS, drho, flag] = sendchange(obj)
            % Return local matrix and vector to be sent to central node
            dS = obj.delta_S_i;
            drho = obj.delta_rho_i;
            TransactionLogger.getInstance().logData('local', obj.S_i, 'send', obj.id);
            TransactionLogger.getInstance().logData('local', obj.rho_i, 'send', obj.id);
            flag = obj.flag;
        end




        function receiveDeltatau(obj, delta_tau, flag, activeset_c)
            % Flag here is 1 if the agent has min dtau and 0 if it doesnt
            if flag == 1
                obj.change_l = true;
                obj.change_c = any(obj.is_active_c ~= activeset_c);   % [FIX] was false
            elseif flag == 0
                obj.change_l = false;
                if any(obj.is_active_c ~= activeset_c)
                    obj.change_c = true;
                else
                    obj.change_c = false;
                end
            end
        
            if isnan(flag)                                   % terminal signal from central
                obj.delta_tau = min(delta_tau, obj.sigma);
                obj.x      = obj.x      + obj.delta_tau * obj.delta_x;
                obj.lambda = obj.lambda + obj.delta_tau * obj.delta_lambda;
                obj.mu     = obj.mu     + obj.delta_tau * obj.delta_mu;
                obj.sigma  = obj.sigma  - obj.delta_tau;     % == 0 bitwise
                obj.tau    = 1 - obj.sigma;
                obj.nu     = obj.nu     + obj.delta_tau * obj.delta_nu;
                mu_s = max(norm(obj.mu, inf), 1);            % ---- snap (terminal) ----
                obj.mu(obj.is_active_l & abs(obj.mu) <= 4*eps*mu_s) = 0;
                obj.mu(~obj.is_active_l) = 0;
                if any(obj.mu < -1e-10*mu_s)
                    error(['node %d: terminal step drove mu to %.3e ' ...
                           '(step = %.3e, sigma = %.3e). The path was closed ' ...
                           'past a dual blocking point; stationarity jump ' ...
                           'would be %.3e.'], ...
                          obj.id, min(obj.mu), obj.delta_tau, obj.sigma, ...
                          norm(obj.H' * min(obj.mu,0), inf));
                end
                obj.mu(obj.mu < 0) = 0;
                return;
            end
        
            obj.delta_tau   = min(delta_tau, obj.sigma);     % handles delta_tau == inf
            obj.is_active_c = activeset_c;
        
            obj.x      = obj.x      + obj.delta_tau * obj.delta_x;
            obj.lambda = obj.lambda + obj.delta_tau * obj.delta_lambda;
            obj.mu     = obj.mu     + obj.delta_tau * obj.delta_mu;
            obj.nu     = obj.nu     + obj.delta_tau * obj.delta_nu;
            obj.sigma  = obj.sigma  - obj.delta_tau;
            obj.tau    = 1 - obj.sigma;
        
            % ---------------- snap ULP residuals to exact zero ----------------
            mu_s = max(norm(obj.mu, inf), 1);
            obj.mu(obj.is_active_l & abs(obj.mu) <= 4*eps*mu_s) = 0;
            obj.mu(~obj.is_active_l) = 0;
        
            if flag == 1
                % Both tests can bind simultaneously (delta_tau_p == delta_tau_d).
                % They must be handled as two independent events, not if/elseif,
                % otherwise mu(k_d) keeps a -ulp residual and is exported.
                did_p = (obj.delta_tau == obj.delta_tau_p) && ~isnan(obj.k_p);
                did_d = (obj.delta_tau == obj.delta_tau_d) && ~isnan(obj.k_d);
        
                % ================= Activate an inequality =================
                if did_p
                    % Check for linear dependance
                    % [ B     G'   H_A' ] [   p  ] = [ H(k_p,:) ]
                    % [ G     0     0   ] [ zeta ] = [     0    ]
                    % [ H_A   0     0   ] [  xi  ] = [     0    ]
                    % Or
                    % [ B   GH' ] [  p   ] = [ H(k_p,:) ]
                    % [ GH   0  ] [  zx  ] = [     0    ]
        
                    QTHi = obj.H(obj.k_p,:)';
                    for j = 1:obj.r_
                        v = zeros(obj.nx,1);
                        v(j:obj.nx) = obj.V(j:obj.nx, j);
                        QTHi(j:obj.nx, :) = QTHi(j:obj.nx, :) - obj.beta(j) * v(j:obj.nx) * (v(j:obj.nx).' * QTHi(j:obj.nx, :));
                    end
                    YHi = QTHi(1:obj.r_,:);
                    ZHi = QTHi(obj.r_+1:obj.nx,:);
        
                    p_y   = obj.Ti' \ zeros(obj.r_,1);
                    rhs_z = ZHi;
        
                    t1  = obj.Li  \ rhs_z;
                    t2  = obj.Di  \ t1;
                    p_z = obj.Li' \ t2;
        
                    p_ = [p_y; p_z];
        
                    % Creating Z* and Y*
                    for j = obj.r_:-1:1
                        v = zeros(obj.nx,1);
                        v(j:obj.nx) = obj.V(j:obj.nx, j);
                        p_(j:obj.nx) = p_(j:obj.nx) - obj.beta(j) * v(j:obj.nx) * (v(j:obj.nx).' * p_(j:obj.nx));
                    end
        
                    rhs_zx = YHi - obj.YBYi * p_y - obj.ZBYi' * p_z;
        
                    zx      = obj.Ti \ rhs_zx;
                    zeta    = zx(1:obj.neq_l,:);
                    xi_temp = zx(obj.neq_l+1:end,:);
                    xi      = obj.mu * 0;
                    xi(obj.is_active_l) = xi_temp;
        
                    tol_dep = sqrt(eps) * max( norm(obj.GHA, 1), ...
                                               norm(obj.H(obj.k_p,:), 1) );
                    if norm(ZHi, 2) <= tol_dep
                        % New active constraint is linearly dependent on previous ones
                        ind_xi = (xi > 0) & obj.is_active_l;
                        if isempty(find(ind_xi, 1))
                            fprintf('The QP is infeasible!\n')
                        else
                            [theta, temp_xi] = min(obj.mu(ind_xi)./xi(ind_xi));
                            active_temp = find(ind_xi);
                            k_ = active_temp(temp_xi);
                        end
        
                        theta = max(theta, 0);                 % clamp: ulp-negative mu
                        obj.lambda = obj.lambda - theta * zeta;
                        obj.mu     = obj.mu     - theta * xi;
                        obj.is_active_l(k_) = false;
                        obj.mu(k_)          = 0;

                        obj.mu(obj.k_p)     = theta;
                    end
        
                    obj.is_active_l(obj.k_p) = true;
                end
        
                % ================= Deactivate an inequality =================
                if did_d
                    if ~did_p
                        % Check for zero curvature after deactivation.
                        % Factorizations below are valid only for the *unmodified*
                        % active set, so this test is skipped when did_p already
                        % changed is_active_l on this step.
                        %
                        % [ B     G'    H_A' ] [   p  ] = [      0     ]
                        % [ G     0      0   ] [ zeta ] = [      0     ]
                        % [ H_A   0      0   ] [  xi  ] = [  -(e_k)_A  ]
                        % Or
                        % [ B   GH' ] [  p   ] = [      0     ]
                        % [ GH   0  ] [  zx  ] = [  -(e_k)_A  ]
        
                        active_idx = find(obj.is_active_l);
                        e_A = zeros(length(active_idx),1);
                        pos = (active_idx == obj.k_d);
                        e_A(pos) = 1;
        
                        rhs_p = [zeros(obj.neq_l,1); -e_A];
                        p_y   = obj.Ti' \ rhs_p;
        
                        rhs_z = - obj.ZBYi * p_y;
        
                        t1  = obj.Li  \ rhs_z;
                        t2  = obj.Di  \ t1;
                        p_z = obj.Li' \ t2;
        
                        p_ = [p_y; p_z];
        
                        % Creating Z* and Y*
                        for j = obj.r_:-1:1
                            v = zeros(obj.nx,1);
                            v(j:obj.nx) = obj.V(j:obj.nx, j);
                            p_(j:obj.nx) = p_(j:obj.nx) - obj.beta(j) * v(j:obj.nx) * (v(j:obj.nx).' * p_(j:obj.nx));
                        end
        
                        r_zx = - obj.YBYi * p_y - obj.ZBYi' * p_z;
                        zx   = obj.Ti \ r_zx;
        
                        if norm(zx, inf) < 1e-14 * max(norm(obj.mu, inf), 1)
                            % B is singular on the null space of the active constraints
                            h_tau_temp = obj.h + obj.sigma * obj.d_h;
        
                            num   = h_tau_temp(~obj.is_active_l,:) - obj.H(~obj.is_active_l,:) * obj.x;
                            den   = obj.H(~obj.is_active_l,:) * p_;
                            den_s = max(norm(den, inf), realmin);
                            ind   = den > 1e-10*den_s;
        
                            if ~any(ind)
                                fprintf('The QP is unbounded along the released direction!\n')
                            else
                                temp = max(num ./ den, 0);
                                [zigma, temp_sigma_ind] = min(temp(ind));
        
                                inactive_indx = find(~obj.is_active_l);
                                valid_indx    = find(ind);
                                sigma_ind     = inactive_indx(valid_indx(temp_sigma_ind));
        
                                if zigma > 1e10
                                    fprintf('The QP is infeasible!\n')
                                else
                                    obj.x = obj.x + zigma * p_;
                                    obj.is_active_l(sigma_ind) = true;
                                end
                            end
                        end
                    end
        
                    obj.is_active_l(obj.k_d) = false;
                    obj.mu(obj.k_d)          = 0;      % unconditional, exact zero
                end
            end
        
            % ---------------- dual sign invariant ----------------
            obj.mu(~obj.is_active_l) = 0;
            mu_s = max(norm(obj.mu, inf), 1);
            if any(obj.mu < -1e-10*mu_s)
                error('node %d: genuine negative multiplier %.3e', obj.id, min(obj.mu));
            end
            obj.mu(obj.mu < 0) = 0;
        
            obj.updatetauvalues();
        


            TransactionLogger.getInstance().logData('local', delta_tau, 'receive', obj.id);
            TransactionLogger.getInstance().logData('local', flag, 'receive', obj.id);
            TransactionLogger.getInstance().logData('local', activeset_c, 'receive', obj.id);
        end


        function [act, mu, lambda, nd, dropped] = resolveDependency(obj, act, mu, lambda)
        % [FIX DEP] Remove rows until Hz(act,:) has full row rank. The leaving
        % row is chosen by a dual ratio test on the dependency null vector, so
        % that  H'*mu + G'*lambda  is left UNCHANGED and mu >= 0 is preserved.
        % Consequence: d_p = p_0 - p is NOT perturbed by the drop (jump == 0),
        % and a member of the dependent group carrying mu = 0 (a B17 row) is
        % removed for free.
        %
        % Rank of A_loc(W,:) modulo range(G') equals rank of Hz(W,:), so the
        % dependency is detected on the reduced rows Hz = H*null(G).
            nd = 0;  dropped = zeros(0,1);  mu0 = mu;  lam0 = lambda;
            tol = 1e-10;

            % rows with no component outside range(G') are unconditionally
            % redundant: H_i is a combination of the equality rows.
            nrm  = vecnorm(obj.Hz, 2, 2);
            dead = act & (nrm <= tol*max(max(nrm), realmin));
            if any(dead)
                mu(dead) = 0;  act(dead) = false;
                dropped  = [dropped; find(dead)];  nd = nd + nnz(dead);
            end

            while true
                W = find(act);
                if numel(W) <= 1
                    break
                end

                Hw = obj.Hz(W,:);
                s  = max(vecnorm(Hw,2,2), realmin);   % row scaling -> scale-free rank test
                D  = null((Hw./s).', tol);
                if isempty(D)
                    break                             % full row rank -> done
                end

                c = D(:,1)./s;                        % Hw.'*c = 0 exactly

                % both +c and -c are legal directions; take the feasible one
                % with the smaller dual step.
                [tp, jp] = LocalNodex0.dualRatio(mu(W),  c);
                [tm, jm] = LocalNodex0.dualRatio(mu(W), -c);
                if tp <= tm
                    t = tp;  j = jp;
                else
                    t = tm;  j = jm;  c = -c;
                end
                if ~isfinite(t)
                    break                             % no sign choice keeps mu >= 0
                end

                muW    = mu(W) - t*c;
                muW(j) = 0;
                mu(W)  = max(muW, 0);                 % clip ULP-level negatives
                act(W(j)) = false;
                dropped   = [dropped; W(j)];
                nd        = nd + 1;
            end

            if nd > 0
                % Hz(W,:)'*(mu-mu0) = 0  =>  H'*(mu-mu0) in range(G'),
                % so this compensation is exact, not least squares:
                %   G'*dlambda = -H'*dmu,  dlambda = -(G G')^{-1} G H' dmu.
                if ~isempty(obj.G)
                    r      = obj.H.' * (mu - mu0);
                    lambda = lambda - obj.GGt \ (obj.G * r);
                end

                jmp = norm(obj.H.'*(mu - mu0) + obj.G.'*(lambda - lam0), inf);
                if jmp > 1e-8*max(norm(obj.H.'*mu0, inf), 1)
                    error('node %d: dependency drop left stationarity jump %.3e', ...
                          obj.id, jmp);
                end
            end
        end


        function xi_out = getdecisionvariable(obj)
            xi_out = obj.x;
        end


        function dual_out = getdual(obj)
            % dual_out = [obj.lambda; obj.mu];
            mu_out = obj.mu;
            mu_out(~obj.is_active_l) = 0;
            mu_out(mu_out < 0) = 0;
            dual_out = [obj.lambda; mu_out];
        end

        function dropcoupled(obj, dropped)
            obj.is_active_c(dropped) = false;
            obj.nu(dropped)       = 0;      % [FIX B10]
            obj.delta_nu(dropped) = 0;      % [FIX B10]
            obj.change_c = true;
            obj.assemblematrices();
        end   

        function active_out = getactiveset(obj)
            % Export W as held. A tight row with mu = 0 is a legitimate member of
            % a degenerate working set; filtering it forces a re-activation next
            % timestep and inflates |W_warm  xor  W*|.
            active_out = obj.is_active_l;
        end

        function residuals = getresiduals(obj)
            res_L = norm( obj.B * obj.x + obj.p + obj.G' * obj.lambda + obj.H' * obj.mu + obj.A' * obj.nu, inf);
            res_eq = norm( obj.G * obj.x - obj.g, inf);
            res_eq_slack = norm( obj.lambda' * (obj.g - obj.G * obj.x), inf);
            sc_h = max(1, norm(obj.h, inf));
            sc_m = max(1, norm(obj.mu, inf));
            res_ineq       = norm(obj.H(obj.is_active_l,:)*obj.x - obj.h(obj.is_active_l), inf) / sc_h;
            res_ineq_viol  = max(0, max(obj.H*obj.x - obj.h)) / sc_h;
            res_ineq_slack = norm(obj.mu .* (obj.h - obj.H*obj.x), inf) / (sc_h*sc_m);
            residuals = [res_L; res_eq; res_eq_slack; res_ineq; res_ineq_viol; res_ineq_slack];
        end

    end


    methods (Static)

        function [t, j] = dualRatio(muW, c)
        % t = min_{j: c_j>0} muW(j)/c(j).  inf if c has no positive
        % entry, i.e. mu_W - t*c cannot be driven to a zero component while
        % keeping mu >= 0 along this direction.
            idx = find(c > 0);
            if isempty(idx)
                t = inf;  j = 0;  return
            end
            [t, k] = min(muW(idx)./c(idx));
            j = idx(k);
        end

    end
end



function [V, beta, R, keptIdx, r] = seqQR_rankreveal(A, tol)
    % A: n x m, columns = active constraint gradients, warm-start order
    % Sequential Householder QR with dependence test before each elimination.
    % A(:,keptIdx) = Q*R,  Q = H_1*H_2*...*H_r  (apply via V,beta)
    
    [n, m] = size(A);
    if nargin < 2
        tol = sqrt(eps) * norm(A,1);
    end
    
    V     = zeros(n,m);
    beta  = zeros(m,1);
    % Q = eye(n);
    R     = zeros(n,m);
    keptIdx = false(1,m);
    k = 0;
    
    for i = 1:m
        a = A(:,i);
        % apply existing reflectors: a <- H_k...H_1*a
        for j = 1:k
            v = V(:,j);
            a = a - beta(j)*(v.'*a)*v;
        end
    
        rho = norm(a(k+1:n));
        if rho <= tol
            continue                 % redundant column -> deactivate constraint i
        end
    
        k = k + 1;
        keptIdx(i) = true;
    
        x = a(k:n);
        s = x(1); if s == 0, s = 1; end
        alpha = -sign(s)*norm(x);
        v = x; v(1) = v(1) - alpha;
        v = v / norm(v);
    
        V(k:n,k) = v;
        beta(k)  = 2;
        R(1:k-1,k) = a(1:k-1);
        R(k,k)     = alpha;
    end
    
    V = V(:,1:k);
    beta = beta(1:k);
    R = R(1:k,1:k);
    r = k;
end


% function x = back_substitute(A, b)
%     % Get the size of the system
%     n = length(b);
% 
%     % Preallocate the solution vector for performance
%     x = zeros(n, 1);
% 
%     % Solve the bottom row first
%     x(n) = b(n) / A(n, n);
% 
%     % Iterate upwards through the matrix
%     for i = n-1:-1:1
%         % A(i, i+1:n) * x(i+1:n) is the vectorized dot product replacing the inner sum
%         x(i) = (b(i) - A(i, i+1:n) * x(i+1:n)) / A(i, i);
%     end
% end