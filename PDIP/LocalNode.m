classdef LocalNode < handle
    %       min_x 1/2 x' Q x + p' x
    % subject to: G x = g
    %             h^{l} <= H x <= h^{u}

    properties %(Access = private)
        id                      % Unique identifier
        Q                       % Quadratic term in local objective function
        p                       % Linear term
        G                       % Local equality constraint matrix
        g                       % Local equality constraint vector
        H                       % Local inequality constraint matrix
        hl                      % Local inequality lower constraint vector
        hu                      % Local inequality upper constraint vector
        A                       % Local matrix of the global coupled constraint
        x                       % Local decision variable
        delta_x                 % Local decision variable step
        nx                      % Dimension of decision variable
        n_eq                    % Number of equality constraints
        n_ineq                  % Number of inequality constraints
        n_Cineq                 % NUmber of coupled inequalities
        lambda_eq               % Lagrange multipliers for equality constraints
        delta_lambda_eq         % Lagrange multipliers change for equality constraints
        lambda_L_ineq           % Lagrange multipliers for lower inequality constraints
        delta_lambda_L_ineq     % Lagrange multipliers change for lower inequality constraints
        lambda_U_ineq           % Lagrange multipliers for upper inequality constraints
        delta_lambda_U_ineq     % Lagrange multipliers change for upper inequality constraints
        mu_l                    % Lagrange multipliers for coupled lower inequality constraints
        mu_u                    % Lagrange multipliers for coupled upper inequality constraints
        Aflag                   % Indicator of existence of lb or ub
        delta_L_mu              % Lagrange multipliers change for coupled lower inequality constraints
        delta_U_mu              % Lagrange multipliers change for coupled upper inequality constraints
        tau                     % Barrier parameter
        r                       % Residual vector
        s                       % Slack variable for lower bound
        t                       % Slack variable for upper bound
        delta_s                 % Slack variable change for lower bound
        delta_t                 % Slack variable change for upper bound
        flag                    % Stopping criterion reached flag = 1
        KKT                     % KKT matrix of local information
        L                       % L matrix of LDL factorization of KKT matrix
        D                       % D matrix of LDL factorization of KKT matrix
        P                       % P (permutation) matrix of LDL factorization of KKT matrix
        Mr                      % KKT^{-1} * RHS_tau
        MA                      % KKT^{-1} * A_bar'
        AKKTr                   % A_extended * KKT^-1 * residual to be communicated 
        AKKTA                   % A_extended * KKT^-1 * A_extended' to be communicated
        I_lower                 % Index of valid lower bounds
        I_upper                 % Index of valid upper bounds
        n_lower                 % Number of lower constraints
        n_upper                 % NUmber of upper constraints
        A_bar                   % Extended matrix A
        tol                     % Stopping tolerance
        delta_z                 % Vector of change for coupled unknowns (mu_l, mu_u, s_c, t_c)
        epsilon                 % Multipler to ensure s + alphga * delta_s >= epsilon * s
        r_d                     % Grad_lag residual
        r_p                     % Residual
        r_s                     % Residual
        r_t                     % Residual
        r_lambda_L_ineq         % Residual    
        r_lambda_U_ineq         % Residual    
    end
    
    methods
        function obj = LocalNode(id, Q, p, G, g, H, hl, hu, A, Aflag, tol)
            obj.epsilon = 0.1;
            obj.id = id;
            obj.Q = Q;
            obj.nx = size(Q,1);                    % Dimension of decision variable 
%             obj.Q = obj.Q + 1e-8*diag(Q);        % Small regulaization for numerical stability
            obj.p = p;
            obj.G = G;
            obj.n_eq = size(G,1);
            obj.n_Cineq = size(A,1);
            obj.g = g;
            obj.H = H;
            obj.n_ineq = size(obj.H,1);
            obj.A = A;
            obj.n_Cineq = size(A,1);
            obj.x = zeros(obj.nx,1);                   % Initialize decision variable
            obj.lambda_eq = zeros(obj.n_eq,1);             % Initialize equality multipliers
            obj.delta_lambda_eq = zeros(obj.n_eq,1);       % Initialize equality multipliers
            obj.tau = 1;
            obj.tol = tol;
            if isempty(hl)
                obj.hl = -inf * ones(obj.n_ineq, 1);
            else
                obj.hl = hl;
            end
            if isempty(hu)
                obj.hu = inf * ones(obj.n_ineq, 1);
            else
                obj.hu = hu;
            end
            if Aflag == 0
                obj.mu_l = ones(obj.n_Cineq,1);
                obj.mu_u = ones(obj.n_Cineq,1);
            elseif Aflag == 1
                obj.mu_l = ones(obj.n_Cineq,1);
                obj.mu_u = [];
            elseif Aflag == -1
                obj.mu_l = [];
                obj.mu_u = ones(obj.n_Cineq,1);
            end
            obj.Aflag = Aflag;

            % Identify active bounds
            obj.I_lower = find(isfinite(hl));
            obj.I_upper = find(isfinite(hu));
            obj.n_lower = length(obj.I_lower);
            obj.n_upper = length(obj.I_upper);
            
            % Initialize variables
            if obj.n_lower > 0
                obj.s = ones(obj.n_lower, 1);                   % Slack variable for lower bound
                obj.delta_s = zeros(obj.n_lower, 1); 
                obj.lambda_L_ineq = - ones(obj.n_lower, 1);     % Lagrange multipliers for lower inequality constraints = ones(n_lower, 1);
            else
                obj.s = [];
                obj.delta_s = [];
                obj.lambda_L_ineq = [];
            end
            if obj.n_upper > 0
                obj.t = ones(obj.n_upper, 1);                   % Slack variable for upper bound
                obj.delta_t = zeros(obj.n_upper, 1);
                obj.lambda_U_ineq = ones(obj.n_upper, 1);       % Lagrange multipliers for upper inequality constraints = ones(n_lower, 1);
            else
                obj.t = [];
                obj.delta_t = [];
                obj.lambda_U_ineq = [];
            end
            obj.flag = 0;
            obj.assemble();

        end

        function obj = assemble(obj)
            % Assemble extended matrix A_bar
            if obj.Aflag == 0
%                 r_d = r_d + obj.A' * obj.mu_l + obj.A' * obj.mu_u;
                if obj.n_lower > 0 && obj.n_upper > 0
                    A_bar = [obj.A',                            obj.A',                             zeros(obj.nx, obj.n_Cineq),         zeros(obj.nx, obj.n_Cineq);
                             zeros(obj.n_eq, obj.n_Cineq),      zeros(obj.n_eq, obj.n_Cineq),       zeros(obj.n_eq, obj.n_Cineq),       zeros(obj.n_eq, obj.n_Cineq);
                             zeros(obj.n_lower, obj.n_Cineq),   zeros(obj.n_lower, obj.n_Cineq),    zeros(obj.n_lower, obj.n_Cineq),    zeros(obj.n_lower, obj.n_Cineq);
                             zeros(obj.n_upper, obj.n_Cineq),   zeros(obj.n_upper, obj.n_Cineq),    zeros(obj.n_upper, obj.n_Cineq),    zeros(obj.n_upper, obj.n_Cineq);
                             zeros(obj.n_lower, obj.n_Cineq),   zeros(obj.n_lower, obj.n_Cineq),    zeros(obj.n_lower, obj.n_Cineq),    zeros(obj.n_lower, obj.n_Cineq);
                             zeros(obj.n_upper, obj.n_Cineq),   zeros(obj.n_upper, obj.n_Cineq),    zeros(obj.n_upper, obj.n_Cineq),    zeros(obj.n_upper, obj.n_Cineq)];
                elseif obj.n_lower > 0
                    A_bar = [obj.A',                            obj.A',                             zeros(obj.nx, obj.n_Cineq),         zeros(obj.nx, obj.n_Cineq);
                             zeros(obj.n_eq, obj.n_Cineq),      zeros(obj.n_eq, obj.n_Cineq),       zeros(obj.n_eq, obj.n_Cineq),       zeros(obj.n_eq, obj.n_Cineq);
                             zeros(obj.n_lower, obj.n_Cineq),   zeros(obj.n_lower, obj.n_Cineq),    zeros(obj.n_lower, obj.n_Cineq),    zeros(obj.n_lower, obj.n_Cineq);
                             zeros(obj.n_lower, obj.n_Cineq),   zeros(obj.n_lower, obj.n_Cineq),    zeros(obj.n_lower, obj.n_Cineq),    zeros(obj.n_lower, obj.n_Cineq)];
                elseif obj.n_upper > 0 
                    A_bar = [obj.A',                            obj.A',                             zeros(obj.nx, obj.n_Cineq),         zeros(obj.nx, obj.n_Cineq);
                             zeros(obj.n_eq, obj.n_Cineq),      zeros(obj.n_eq, obj.n_Cineq),       zeros(obj.n_eq, obj.n_Cineq),       zeros(obj.n_eq, obj.n_Cineq);
                             zeros(obj.n_upper, obj.n_Cineq),   zeros(obj.n_upper, obj.n_Cineq),    zeros(obj.n_upper, obj.n_Cineq),    zeros(obj.n_upper, obj.n_Cineq);
                             zeros(obj.n_upper, obj.n_Cineq),   zeros(obj.n_upper, obj.n_Cineq),    zeros(obj.n_upper, obj.n_Cineq),    zeros(obj.n_upper, obj.n_Cineq)];
                end

            elseif obj.Aflag == 1 || obj.Aflag == -1
%                 r_d = r_d + obj.A' * obj.mu_l;
                if obj.n_lower > 0 && obj.n_upper > 0
                    A_bar = [obj.A',                            zeros(obj.nx, obj.n_Cineq);
                             zeros(obj.n_eq, obj.n_Cineq),      zeros(obj.n_eq, obj.n_Cineq);
                             zeros(obj.n_lower, obj.n_Cineq),   zeros(obj.n_lower, obj.n_Cineq);
                             zeros(obj.n_upper, obj.n_Cineq),   zeros(obj.n_upper, obj.n_Cineq);
                             zeros(obj.n_lower, obj.n_Cineq),   zeros(obj.n_lower, obj.n_Cineq);
                             zeros(obj.n_upper, obj.n_Cineq),   zeros(obj.n_upper, obj.n_Cineq)]';
                elseif obj.n_lower > 0
                    A_bar = [obj.A',                            zeros(obj.nx, obj.n_Cineq);
                             zeros(obj.n_eq, obj.n_Cineq),      zeros(obj.n_eq, obj.n_Cineq);
                             zeros(obj.n_lower, obj.n_Cineq),   zeros(obj.n_lower, obj.n_Cineq);
                             zeros(obj.n_lower, obj.n_Cineq),   zeros(obj.n_lower, obj.n_Cineq)]';
                elseif obj.n_upper > 0 
                    A_bar = [obj.A',                            zeros(obj.nx, obj.n_Cineq);
                             zeros(obj.n_eq, obj.n_Cineq),      zeros(obj.n_eq, obj.n_Cineq);
                             zeros(obj.n_upper, obj.n_Cineq),   zeros(obj.n_upper, obj.n_Cineq);
                             zeros(obj.n_upper, obj.n_Cineq),   zeros(obj.n_upper, obj.n_Cineq)]';
                end

            end

            % Compute residuals
            if ~isempty(obj.G)
                obj.r_d = obj.Q * obj.x + obj.p + obj.G' * obj.lambda_eq;
            else
                obj.r_d = obj.Q * obj.x + obj.p;
            end
            if obj.n_lower > 0 && obj.n_upper > 0
                obj.r_d = obj.r_d + obj.H(obj.I_lower, :)' * obj.lambda_L_ineq + obj.H(obj.I_upper, :)' * obj.lambda_U_ineq;
            elseif obj.n_lower > 0
                obj.r_d = obj.r_d + obj.H(obj.I_lower, :)' * obj.lambda_L_ineq;
            elseif obj.n_upper > 0
                obj.r_d = obj.r_d + obj.H(obj.I_upper, :)' * obj.lambda_U_ineq;
            end
            
            if ~isempty(obj.G)
                obj.r_p = obj.G * obj.x - obj.g;
            end
            if obj.n_lower > 0
                obj.r_s = obj.H(obj.I_lower, :) * obj.x - obj.s - obj.hl(obj.I_lower);
                obj.r_lambda_L_ineq = diag(obj.s) * obj.lambda_L_ineq - obj.tau * ones(obj.n_lower, 1);
            end
            if obj.n_upper > 0
                obj.r_t = obj.H(obj.I_upper, :) * obj.x + obj.t - obj.hu(obj.I_upper);
                obj.r_lambda_U_ineq = diag(obj.t) * obj.lambda_U_ineq - obj.tau * ones(obj.n_upper, 1);
            end

            if obj.n_lower > 0 && obj.n_upper > 0
                obj.KKT = [obj.Q,                       obj.G',                         obj.H(obj.I_lower, :)',            obj.H(obj.I_upper, :)'             zeros(obj.nx, obj.n_lower),       zeros(obj.nx, obj.n_upper);
                           obj.G,                       zeros(obj.n_eq),                zeros(obj.n_eq, obj.n_lower),      zeros(obj.n_eq, obj.n_upper),      zeros(obj.n_eq, obj.n_lower),     zeros(obj.n_eq, obj.n_upper);
                           obj.H(obj.I_lower, :),       zeros(obj.n_lower, obj.n_eq),   zeros(obj.n_lower),                zeros(obj.n_lower, obj.n_upper),   -eye(obj.n_lower),                zeros(obj.n_lower, obj.n_upper);
                           obj.H(obj.I_upper, :),       zeros(obj.n_upper, obj.n_eq),   zeros(obj.n_upper, obj.n_lower),   zeros(obj.n_upper),                zeros(obj.n_upper, obj.n_lower),  eye(obj.n_upper);
                           zeros(obj.n_lower, obj.nx),  zeros(obj.n_lower, obj.n_eq),   diag(obj.s),                       zeros(obj.n_lower, obj.n_upper),   diag(obj.lambda_L_ineq),          zeros(obj.n_lower, obj.n_upper);
                           zeros(obj.n_upper, obj.nx),  zeros(obj.n_upper, obj.n_eq),   zeros(obj.n_upper, obj.n_lower),   diag(obj.t),                       zeros(obj.n_upper, obj.n_lower),  diag(obj.lambda_U_ineq)];
                
                obj.r = - [obj.r_d; obj.r_p; obj.r_s; obj.r_t; obj.r_lambda_L_ineq; obj.r_lambda_U_ineq];

            elseif obj.n_lower > 0
                obj.KKT = [obj.Q,                       obj.G',                         obj.H(obj.I_lower, :)',            zeros(obj.nx, obj.n_lower);
                           obj.G,                       zeros(obj.n_eq),                zeros(obj.n_eq, obj.n_lower),      zeros(obj.n_eq, obj.n_lower);
                           obj.H(obj.I_lower, :),       zeros(obj.n_lower, obj.n_eq),   zeros(obj.n_lower),                -eye(obj.n_lower);
                           zeros(obj.n_lower, obj.nx),  zeros(obj.n_lower, obj.n_eq),   diag(obj.s),                       diag(obj.lambda_L_ineq);];
                
                obj.r = - [obj.r_d; obj.r_p; obj.r_s; obj.r_lambda_L_ineq];

            elseif obj.n_upper > 0
                obj.KKT = [obj.Q,                       obj.G',                         obj.H(obj.I_upper, :)'             zeros(obj.nx, obj.n_upper);
                           obj.G,                       zeros(obj.n_eq),                zeros(obj.n_eq, obj.n_upper),      zeros(obj.n_eq, obj.n_upper);
                           obj.H(obj.I_upper, :),       zeros(obj.n_upper, obj.n_eq),   zeros(obj.n_upper),                eye(obj.n_upper);
                           zeros(obj.n_upper, obj.nx),  zeros(obj.n_upper, obj.n_eq),   diag(obj.t),                       diag(obj.lambda_U_ineq)];
                
                obj.r = - [obj.r_d; obj.r_p; obj.r_t; obj.r_lambda_U_ineq];
            else
                obj.KKT = [obj.Q, obj.G'; obj.G, zeros(obj.n_eq)];
                obj.r = - [obj.r_d; obj.r_p];
            end
            obj.A_bar = A_bar;
            obj.MA = obj.KKT \ obj.A_bar';
            obj.Mr = obj.KKT \ obj.r;
        end

                
        function [AKKTAT, AKKTr, flag] = sendDataToCentral(obj)
            % Return local matrix and vector to be sent to central node
            AKKTAT = obj.A_bar * obj.MA;
            AKKTr = obj.A_bar * (obj.Mr);
            if norm(obj.r, inf) <= obj.tol
                flag = 1;
            else 
                flag = 0;
            end
        end

        function receiveDeltaz(obj, delta_z)
            % Store received delta_mu from central node
            obj.delta_z = delta_z;
            dir = obj.Mr - obj.MA * delta_z;
            obj.delta_x = dir(1 : obj.nx);
            obj.delta_lambda_eq = dir(obj.nx + 1 : obj.nx + obj.n_eq);
            obj.delta_lambda_L_ineq = dir( obj.nx + obj.n_eq + 1 : obj.nx + obj.n_eq + obj.n_lower);
            obj.delta_lambda_U_ineq = dir( obj.nx + obj.n_eq + obj.n_lower + 1 : obj.nx + obj.n_eq + obj.n_lower + obj.n_upper);
            obj.delta_s = dir(obj.nx + obj.n_eq + obj.n_lower + obj.n_upper + 1 : obj.nx + obj.n_eq + obj.n_lower + obj.n_upper + obj.n_lower);
            obj.delta_t = dir(obj.nx + obj.n_eq + obj.n_lower + obj.n_upper + obj.n_lower + 1 : obj.nx + obj.n_eq + obj.n_lower + obj.n_upper + obj.n_lower + obj.n_upper);
        end

        
        function [f, gradfT_deltax, Gx, Hxl, Hxu, Ax, alpha] = sendDataToCentralstep(obj, alpha_)
            if obj.n_lower > 0
                alpha_s = min(1 , (- 0.9 * obj.s) ./ obj.delta_s);
            else
                alpha_s = 1e6;
            end
            if obj.n_upper > 0
                alpha_t = min([1 ; (- 0.9 * obj.t) ./ obj.delta_t]);
            else
                alpha_t = 1e6;
            end
            alpha = min(alpha_s, alpha_t);
            f = 1/2 * (obj.x + alpha_ * obj.delta_x)' * obj.Q * (obj.x + alpha_ * obj.delta_x) + obj.p' * (obj.x + alpha_ * obj.delta_x);
            gradfT_deltax = obj.x' * obj.Q * obj.delta_x + obj.p' * obj.delta_x;
            if ~isempty(obj.G)
                Gx = obj.G * (obj.x + alpha_ * obj.delta_x) - obj.g;
                Gx = norm(Gx, 1);
            end
            Ax = obj.A * (obj.x + alpha_ * obj.delta_x);
            if obj.n_lower > 0
%                 Hxl = max(obj.hl - obj.H * (obj.x + alpha_ * obj.delta_x) , 0);
                Hxl = obj.hl - obj.H * (obj.x + alpha_ * obj.delta_x) + (obj.s + alpha_ * obj.delta_s);
                Hxl = norm(Hxl, 1);
            else 
                Hxl = 0;
            end
            if obj.n_upper > 0
%                 Hxu = max(obj.H * (obj.x + alpha_ * obj.delta_x) -obj.hu, 0);
                Hxu = obj.H * (obj.x + alpha_ * obj.delta_x) + (obj.t + alpha_ * obj.delta_t) - obj.hu;
                Hxu = norm(Hxu, 1);
            else 
                Hxu = 0;
            end
        end
        
        function receivealpha(obj, alpha)
            obj.x = obj.x + alpha * obj.delta_x;
            obj.lambda_eq = obj.lambda_eq + alpha * obj.delta_lambda_eq;
            obj.lambda_L_ineq = obj.lambda_L_ineq + alpha * obj.delta_lambda_L_ineq;
            obj.lambda_U_ineq = obj.lambda_U_ineq + alpha * obj.delta_lambda_U_ineq;
            obj.s = obj.s + alpha * obj.delta_s;
            obj.t = obj.t + alpha * obj.delta_t;   
            obj.assemble();
        end

        function [r, Ax] = sendresidual(obj)
            if obj.n_lower > 0 && obj.n_upper > 0
               r = - [obj.r_d; obj.r_p; obj.r_s; obj.r_t; obj.r_lambda_L_ineq; obj.r_lambda_U_ineq];
            elseif obj.n_lower > 0
               r = - [obj.r_d; obj.r_p; obj.r_s; obj.r_lambda_L_ineq];
            elseif obj.n_upper > 0
               r = - [obj.r_d; obj.r_p; obj.r_t; obj.r_lambda_U_ineq];
            else
               r = - [obj.r_d; obj.r_p];
            end
            Ax = obj.A * obj.x;            
        end

        function xi_out = getdecisionvariable(obj)
            xi_out = obj.x;
        end
    end
end
