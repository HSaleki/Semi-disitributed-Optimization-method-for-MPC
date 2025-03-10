classdef LocalNode < handle
    %       min_x 1/2 x' Q x + p' x
    % subject to: G x = g
    %             h^{l} <= H x <= h^{u}

    properties (Access = private)
        id                      % Unique identifier
        Q                       % Quadratic term in local objective function
        p                       % Linear term
        G                       % Local equality constraint matrix
        g                       % Local equality constraint vector
        H                       % Local inequality constraint matrix
        hl                      % Local inequality lower constraint vector
        hu                      % Local inequality upper constraint vector
        A                       % Local matrix of the global coupled constraint
        Aw                      % Local matrix of the active global coupled constraint
        W_c                     % Working set of the global coupled constraint 
        x                       % Local decision variable
        delta_x                 % Local decision variable step
        nx                      % Dimension of decision variable
        nineq                   % Number of inequality constraints
        lambda_eq               % Lagrange multipliers for equality constraints
        delta_lambda_eq         % Lagrange multipliers change for equality constraints
        lambda_ineq             % Lagrange multipliers for inequality constraints
        delta_lambda_ineq       % Lagrange multipliers change for inequality constraints
        delta_mu                % Lagrange multipliers change for coupled inequality constraints
        W                       % Working set (W_i = -1 for lower active, 0 inactive, +1 for upper active constraints)
        activeset_l             % Active set for local inequality constraint
        tau                     % Homotopy parameter   
        delta_tau               % Change in tau in each iteration
        Hw                      % Active inequality matrix
        hw                      % Active inequality vector
        hw_tau                  % Active inequality vector at tau
        r                       % Perturbation matrix
        hl_tau                  % Lower bound at tau
        hu_tau                  % Upper bound at tau
        p_tau                   % Gradeitn vector at tau
        g_tau                   % Equality constraint at tau    
        hl_0                    % Initial lower bound
        hu_0                    % Initial upper bound
        p_0                     % Initial gradient vector
        g_0                     % Initial equality constraint vector
        Delta1                  % Vector of final gradient and bounds for stopping criterion
        DeltaTau                % Vector of gradient and bounds at tau for stopping criterion
        flag                    % Stopping criterion reached flag = 1
        KKT                     % KKT matrix of local information
        L                       % L matrix of LDL factorization of KKT matrix
        D                       % D matrix of LDL factorization of KKT matrix
        P                       % P (permutation) matrix of LDL factorization of KKT matrix
        Mr                      % KKT^{-1} * RHS_tau
        MA                      % KKT^{-1} * A_bar'
        AKKTr                   % A_extended * KKT^-1 * residual to be communicaeted 
        AKKTA                   % A_extended * KKT^-1 * A_extended' to be communicaeted
        delta_tau_p             % delta_tau from primal ratio
        delta_tau_da            % delta_tau from dual ratio
        delta_tau_w             % delta_tau from dual ratio for weakly active constraints
        l                       % Index of inactive constraint becoming active
        lw                      % Index of active constraint with sign change
        idx
        ka
        W_new
    end
    
    methods
        function obj = LocalNode(id, Q, p, G, g, H, hl, hu, A)
            obj.id = id;
            obj.Q = Q;
            obj.nx = size(Q,1);
            obj.Q = obj.Q + 1e-8*diag(Q);        % Small regulaization for numerical stability
            obj.p = p;
            obj.G = G;
            obj.g = g;
            obj.H = H;
            obj.nineq = size(obj.H,1);
            obj.hl = assignIfEmpty(hl, obj.nineq, -1);
            obj.hu = assignIfEmpty(hu, obj.nineq, 1);
            obj.A = A;
            obj.x = zeros(obj.nx,1);                   % Initialize decision variable
            obj.lambda_eq = zeros(size(obj.g));             % Initialize equality multipliers
            obj.delta_lambda_eq = zeros(size(obj.g));       % Initialize equality multipliers
            obj.lambda_ineq = zeros(obj.nineq,1);         % Initialize inequality multipliers
            obj.delta_lambda_ineq = zeros(obj.nineq,1);   % Initialize inequality multipliers
            obj.W = zeros(obj.nineq,1);                   % Initialize will all inequalities inactive
            obj.W_c = zeros(size(obj.A,1));
            obj.flag = 0;
            obj.tau = 0;
            obj.delta_tau = 0;
            obj.delta_mu = zeros(size(A,1),1);
            obj.activeset_l = find(obj.W ~= 0);
            obj.p_0 = zeros(obj.nx,1);
            obj.hl_0 = zeros(obj.nineq,1);
            obj.hu_0 = zeros(obj.nineq,1);
            obj.Initialize();
            obj.updatetauvalues();
            obj.assembleKKT();
            obj.W_new = obj.W;
        end

        function obj = Initialize(obj)
            if ~isempty(obj.G)
                obj.Delta1 = [obj.p; obj.g; obj.hl; obj.hu];
            else
                obj.Delta1 = [obj.p; obj.hl; obj.hu];
            end
            lb_0 = zeros(obj.nineq,1);
            ub_0 = zeros(obj.nineq,1);
            constraints = 1:obj.nineq;
            if obj.nineq > 1
                r_initial = 0.5;
                r_final = 1;
                r_values = r_initial * (obj.nineq - constraints) / (obj.nineq - 1) + r_final * (constraints - 1) / (obj.nineq - 1);
                obj.r = repmat(r_values, 3, 1);
    
                % Perturb Lagrange multipliers
                obj.lambda_ineq(obj.W == -1) = obj.lambda_ineq(obj.W == -1) + obj.r(1, obj.W == -1)'; % Lower bound active
                obj.lambda_ineq(obj.W == +1) = obj.lambda_ineq(obj.W == +1) - obj.r(1, obj.W == +1)'; % Upper bound active
                % Compute perturbed lower bounds hl(0)
                obj.hl_0(obj.W ~= -1) = lb_0(obj.W ~= -1) - obj.r(2, obj.W ~= -1)'; % Apply perturbation only at tau = 0
                % Compute perturbed lower bounds hu(0)    
                obj.hu_0(obj.W ~= 1) = ub_0(obj.W ~= 1) + obj.r(3, obj.W ~= 1)';    % Apply perturbation only at tau = 0
            end
            % Compute adapted gradient p(0)
            if ~isempty(obj.G)
                obj.p_0 = obj.G' * obj.lambda_eq + obj.H' * obj.lambda_ineq - obj.Q * obj.x;
            else
                obj.p_0 = obj.H' * obj.lambda_ineq - obj.Q * obj.x;
            end
            obj.g_0 = zeros(size(obj.g));
        end

        function obj = updatetauvalues(obj)
%             obj.delta_tau = delta_tau;
%             obj.tau = obj.tau + obj.delta_tau;
            % Compute adaptive bounds and gradient
            obj.p_tau = (1 - obj.tau) * obj.p_0 + obj.tau * obj.p;
            obj.hl_tau = (1 - obj.tau) * obj.hl_0 + obj.tau * obj.hl;
            obj.hu_tau = (1 - obj.tau) * obj.hu_0 + obj.tau * obj.hu;
            obj.g_tau = (1 - obj.tau) * obj.g_0 + obj.tau * obj.g;
            % Return DeltaTau as a concatenated vector
            if ~isempty(obj.G)
                obj.DeltaTau = [obj.p_tau; obj.g_tau; obj.hl_tau; obj.hu_tau];
            else
                obj.DeltaTau = [obj.p_tau; obj.hl_tau; obj.hu_tau];
            end
            denom = max(abs(obj.DeltaTau), abs(obj.Delta1)); % Element-wise max
            idxe = denom < 1e-15;  % Logical index where denom is smaller than tol
            
            s = (obj.DeltaTau - obj.Delta1) ./ denom; % Compute s element-wise
            s(idxe) = 0;  % Set values where denom < tol to 0 
            
            delta = norm(s, Inf);
            if delta <= 1e-15
                obj.flag = 1;
            end
            obj.assembleKKT();

        end

        function obj = assembleKKT(obj)
            % Assebmle matrices and vectors required for KKT system.
            % factorize and solve it
            % Extract rows of A corresponding to currently active constraints
            k = size(obj.G);
            nA = length(obj.activeset_l);
            % Initialize vector of bounds of active constrains
            obj.hw_tau = zeros(nA,1);
            obj.hw = zeros(nA,1);

            if ~isempty(obj.activeset_l)
                obj.Hw = obj.H(obj.activeset_l,:);

                % Build the KKT system for current tau
                %   [  Q     G'     Hw'  ]
                %   [  G     0       0   ]
                %   [  Hw    0       0   ]
                %
                obj.KKT = [obj.Q,   obj.G',     obj.Hw';...
                       obj.G,   zeros(k),   zeros(k(1),nA);...
                       obj.Hw,  zeros(nA,k(2)),    zeros(nA)];

                idx_neg = obj.W(obj.activeset_l) == -1;
                idx_pos = obj.W(obj.activeset_l) == 1;
                
                obj.hw_tau(idx_neg) = obj.hl_tau(obj.activeset_l(idx_neg));
                obj.hw(idx_neg) = obj.hl(obj.activeset_l(idx_neg));
                
                obj.hw_tau(idx_pos) = obj.hu_tau(obj.activeset_l(idx_pos));
                obj.hw(idx_pos) = obj.hu(obj.activeset_l(idx_pos));

                % Build the right-hand side
                %   [  - ( p((1) - p(tau) )  ]
                %   [      g(1) - g(tau)     ]
                %   [     hw(1) - hw(tau)    ]
                %
                RHS = [-( obj.p - obj.p_tau);...
                         obj.g - obj.g_tau;...
                        obj.hw - obj.hw_tau];

            else
                % Build the KKT system for current tau
                %   [  Q     G'  ]
                %   [  G     0   ]
                %
                obj.KKT = [obj.Q,   obj.G';...
                           obj.G,   zeros(k);];

                % Build the right-hand side
                %   [  - ( p((1) - p(tau) )  ]
                %   [      g(1) - g(tau)     ]
                %
                RHS = [-( obj.p - obj.p_tau);...
                         obj.g - obj.g_tau];

            end

            %--------------------------------------------------------------

            [obj.L, obj.D] = ldl(obj.KKT);   % LDL^T factorization
            activeset_c = find(obj.W_c ~= 0);
            nCA = length(activeset_c);
            if nCA == 0
                obj.Aw = [];
                obj.AKKTr = [];
                obj.AKKTA = [];
                obj.MA = [];
                
                y1 = obj.L \ RHS;
                z1 = obj.D \ y1;
                obj.Mr = obj.L' \ z1;       % KKT^{-1} * RHS_tau
            else
                obj.Aw = obj.A(activeset_c,:);
            
                A_bar = [obj.Aw; zeros(k(1),nCA); zeros(nCA)];
                % Calculate A_bar * KKT^-1 * RHS
                y1 = obj.L \ RHS;
                z1 = obj.D \ y1;
                obj.Mr = obj.L' \ z1;       % KKT^{-1} * RHS_tau
                
                obj.AKKTr = A_bar * obj.Mr;
                
                y2 = obj.L \ A_bar';
                z2 = obj.D \ y2;
                obj.MA = obj.L' \ z2;       % KKT^{-1} * A_bar'
    
                obj.AKKTA = A_bar * obj.MA;
            end


        end
        
        function [AKKTAT, AKKTr, flag] = sendDataToCentral(obj)
            % Return local matrix and vector to be sent to central node
            AKKTAT = obj.AKKTA;
            AKKTr = obj.AKKTr;
            flag = obj.flag;
        end

        function receiveDeltaMu(obj, delta_mu)
            % Store received delta_mu from central node
            obj.delta_mu = delta_mu;
            if obj.W_c == 0
                delta_z = obj.Mr;
            else
                delta_z = obj.Mr + obj.MA * obj.delta_mu;
            end
            delta_z = delta_z / (1 - obj.tau);
            obj.delta_x = delta_z(1:obj.nx);
            obj.delta_lambda_eq = - delta_z(obj.nx + 1: obj.nx + (size(obj.g)));
            delta_lambda_temp = - delta_z(obj.nx + (size(obj.g)) + 1: end);
            obj.delta_lambda_ineq(obj.activeset_l) = delta_lambda_temp;
            obj.homotopystep();
        end

        function obj = homotopystep(obj)
            InactiveInequalities = setdiff(1:obj.nineq, obj.activeset_l);
            H_inactive = obj.H(InactiveInequalities,:);

            hl_inactive_tau = obj.hl_tau(InactiveInequalities);
            hu_inactive_tau = obj.hu_tau(InactiveInequalities);
            

            [temp(1,1), temp(1,2)] = RT( H_inactive * obj.x - hl_inactive_tau, obj.hl(InactiveInequalities) - obj.hl_0(InactiveInequalities) - H_inactive * obj.delta_x, 1e-15, 1e-15, InactiveInequalities);
            [temp(2,1), temp(2,2)] = RT( hu_inactive_tau - H_inactive * obj.x, H_inactive * obj.delta_x + obj.hu_0(InactiveInequalities) - obj.hu(InactiveInequalities), 1e-15, 1e-15, InactiveInequalities);
 
            [obj.delta_tau_p, obj.idx] = min(temp(:,1));
            obj.l = temp(obj.idx,2);

            % Among active constraints, see which one is violated most
            [obj.delta_tau_da, obj.ka] =  RT( - obj.W .* obj.lambda_ineq, obj.W .* obj.delta_lambda_ineq, 1e-15, 1e-15, []);

            % If a constraint is weakly active, treat it like inactive
            WeakIdx = find(obj.W~= 0 & abs(obj.lambda_ineq) < 1e-15);
            H_weak = obj.H(WeakIdx,:);
            hl_weak_tau = obj.hl_tau(WeakIdx);
            hu_weak_tau = obj.hu_tau(WeakIdx);
            [temp2(1,1), temp2(1,2)] = RT( H_weak * obj.x - hl_weak_tau, - H_weak * obj.delta_x, 1e-15, 1e-15, WeakIdx);
            [temp2(2,1), temp2(2,2)] = RT( hu_weak_tau - H_weak * obj.x, H_weak * obj.delta_x, 1e-15, 1e-15, WeakIdx);
            
            [obj.delta_tau_w, idx2] = min(temp2(:,1));
            obj.lw = temp2(idx2,2);
            
            % Choose smallest valid step
            obj.delta_tau = min([obj.delta_tau_p, obj.delta_tau_da, obj.delta_tau_w]);

        end
        
        function [Axi, Adeltaxi, delta_taui] = sendDataToCentralstep(obj)
            % Return local matrix and vector to be sent to central node
            inactivecoupled = setdiff(1:size(obj.A,1), find(obj.W_c ~= 0));
            A_inactive = obj.A(inactivecoupled,:);
            Axi = A_inactive * obj.x;
            Adeltaxi = A_inactive * obj.delta_x;
            delta_taui = obj.delta_tau;
        end
        
        function receiveDeltatau(obj, delta_tau, flag, Wc)

            if (obj.delta_tau >= 1 - obj.tau) || delta_tau == inf
                
            else
                if flag == 1
                    if obj.delta_tau == obj.delta_tau_p
                      if obj.idx == 1
                          obj.W_new(obj.l) = -1;
                      elseif obj.idx == 2
                          obj.W_new(obj.l) = 1;
                      end
                    elseif obj.delta_tau == obj.delta_tau_da
                        if ~isnan(obj.ka)
                            obj.W_new(obj.ka) = 0;
                        end
                    end
                end
            end

            obj.delta_tau = delta_tau;
            
            if (obj.delta_tau >= 1 - obj.tau) || delta_tau == inf
                obj.x = obj.x + (1 - obj.tau) * obj.delta_x;
                obj.lambda_eq = obj.lambda_eq + (1 - obj.tau) * obj.delta_lambda_eq;
                obj.lambda_ineq = obj.lambda_ineq + (1 - obj.tau) * obj.delta_lambda_ineq;
                obj.tau = 1;
            else
                obj.tau = obj.tau + obj.delta_tau;
                obj.x = obj.x + obj.delta_tau * obj.delta_x;
                obj.lambda_eq = obj.lambda_eq + obj.delta_tau * obj.delta_lambda_eq;
                obj.lambda_ineq = obj.lambda_ineq + obj.delta_tau * obj.delta_lambda_ineq;
            end
            
            obj.W = obj.W_new;
            obj.W_c = Wc;
            obj.activeset_l = find(obj.W ~= 0);
            obj.updatetauvalues();
        end

        function xi_out = getdecisionvariable(obj)
            xi_out = obj.x;
        end
    end
end