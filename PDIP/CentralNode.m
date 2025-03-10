classdef CentralNode < handle
    properties
        nodes           % Array of LocalNode instances
        mu              % Global Lagrange multiplier
        delta_mu        % Global Lagrange multiplier change
        lb              % Global constraint lower bound vector
        ub              % Global constraint upper bound vector
        mu_l            % Global Lagrange multiplier for lower coupled constraint 
        mu_u            % Global Lagrange multiplier for upper coupled constraint
        s_c             % Slack variable for lower coupled constraint
        t_c             % Slack variable for upper coupled constraint
        delta_s_c       % Slack variable change for lower coupled constraint
        delta_t_c       % Slack variable change for upper coupled constraint
        delta_mu_l      % Global Lagrange multiplier change for lower coupled constraint 
        delta_mu_u      % Global Lagrange multiplier change for upper coupled constraint
        n_Cineq         % Number of inequality constraints (size(A,1))
        r               % residual vector
        tau             % Homotopy parameter
        delta_tau       % Homotopy parameter step
        Ax_sum          % Sum_i A_i * x_i   
        Adelta_x_sum    % Sum_i A_i * delta_x_i
        delta_taus      % Vector of delta_tau from all local nodes and central one
        stopflag        % Flag for stopping when reached the accuracy
        flagc           % Flag for reaching coupled constrained accuracy
        exit_flag       % Exit flag for main script
        tol             % Stopping tolerance
        Aflag           % Indicator of existence of lb or ub
        M               % Corresponding matrix of coupled constraints
        z               % Vector for coupled unknowns (mu_l, mu_u, s_c, t_c)
        delta_z         % Vector of change for coupled unknowns (mu_l, mu_u, s_c, t_c)
        sigma           % multiplier in merit function
        M1              % M1 merit function value
        DM1             % Directional derivative of M1 merit function value
        alpha           % Step length
    end
    
    methods
        function obj = CentralNode(nodes, lb, ub, tol)
            obj.nodes = nodes;
            obj.n_Cineq = max(length(lb), length(ub));
            if ~isempty(lb) && ~isempty(ub)
                obj.mu_l = ones(obj.n_Cineq,1);
                obj.mu_u = ones(obj.n_Cineq,1);
                obj.delta_mu_l = zeros(obj.n_Cineq,1);
                obj.delta_mu_u = zeros(obj.n_Cineq,1);
                obj.s_c = ones(obj.n_Cineq,1);
                obj.t_c = ones(obj.n_Cineq,1);
                obj.delta_s_c = zeros(obj.n_Cineq,1);
                obj.delta_t_c = zeros(obj.n_Cineq,1);
                obj.Aflag = 0;
                obj.lb = lb;
                obj.ub = ub;
            elseif isempty(ub)
                obj.mu_l = - ones(obj.n_Cineq,1);
                obj.mu_u = [];
                obj.delta_mu_l = zeros(obj.n_Cineq,1);
                obj.s_c = ones(obj.n_Cineq,1);
                obj.delta_s_c = zeros(obj.n_Cineq,1);
                obj.lb = lb;
                obj.ub = [];
                obj.Aflag = 1;
            elseif isempty(lb)
                obj.mu_l = [];
                obj.mu_u = ones(obj.n_Cineq,1);
                obj.delta_mu_u = zeros(obj.n_Cineq,1);
                obj.t_c = ones(obj.n_Cineq,1);
                obj.delta_t_c = zeros(obj.n_Cineq,1);
                obj.lb = [];
                obj.ub = ub;
                obj.Aflag = -1;
            end
            obj.tau = 1;
            obj.delta_tau = 0;
            obj.Ax_sum = zeros(obj.n_Cineq,1);
            obj.exit_flag = 0;
            obj.stopflag = zeros(length(obj.nodes)+1,1);
            obj.assemble();
            obj.flagc = 0;
            obj.tol = tol;
            obj.sigma = 2;
        end

       
        function obj = assemble(obj)
            if obj.Aflag == 0
                obj.M = [zeros(obj.n_Cineq),    zeros(obj.n_Cineq),     - ones(obj.n_Cineq),    zeros(obj.n_Cineq);
                         zeros(obj.n_Cineq),    zeros(obj.n_Cineq),     zeros(obj.n_Cineq),     ones(obj.n_Cineq);
                         diag(obj.s_c),         ones(obj.n_Cineq),      diag(obj.mu_l),         ones(obj.n_Cineq);
                         ones(obj.n_Cineq),     diag(obj.t_c),          ones(obj.n_Cineq),      diag(obj.mu_u)];
                obj.r = [obj.Ax_sum - obj.s_c - obj.lb;
                         obj.Ax_sum + obj.t_c - obj.ub;
                         diag(obj.s_c) * obj.mu_l - obj.tau * ones(obj.n_Cineq,1);
                         diag(obj.t_c) * obj.mu_u - obj.tau * ones(obj.n_Cineq,1)];
            elseif obj.Aflag == 1
                obj.M = [zeros(obj.n_Cineq),    - ones(obj.n_Cineq);
                         diag(obj.s_c),         diag(obj.mu_l)];
                obj.r = [obj.Ax_sum - obj.s_c - obj.lb;
                         diag(obj.s_c) * obj.mu_l - obj.tau * ones(obj.n_Cineq,1)];
            elseif obj.Aflag == -1
                obj.M = [zeros(obj.n_Cineq),     ones(obj.n_Cineq);
                         diag(obj.t_c),          diag(obj.mu_u)];
                obj.r = [obj.Ax_sum + obj.t_c - obj.ub;
                         diag(obj.t_c) * obj.mu_u - obj.tau * ones(obj.n_Cineq,1)];
            end
            if norm(obj.r, inf) <= obj.tol
                obj.flagc = 1;
            end
        end

        function delta_z = aggregateAndSolveDeltaz(obj)
            % Initialize accumulators
            M_sum = 0;
            v_sum = 0;
            
            % Aggregate data from all local nodes
            for i = 1:length(obj.nodes)
                [M_i, v_i, stopflag_i] = obj.nodes{i}.sendDataToCentral(); % Receive A_bar KKT^-1 A_bar', A_bar KKT^-1 r 
                M_sum = M_sum + M_i;
                v_sum = v_sum + v_i;
                obj.stopflag(i) = stopflag_i;
                obj.stopflag(length(obj.nodes)+1) = obj.flagc;
            end
            
            % Solve for delta_z: (sum_i M_i) * delta_mu = r - (sum_i v_i)
            % delta_z = [delta_mu_l; delta_s_cc; delta_mu_u; delta_t_c];
            delta_z = (obj.M - M_sum) \ (obj.r - v_sum);
            if obj.Aflag == 0
                obj.delta_mu_l = delta_z(1:obj.n_Cineq);
                obj.delta_mu_u = delta_z(obj.n_Cineq+1:2*obj.n_Cineq);
                obj.delta_s_c = delta_z(2*obj.n_Cineq+1:3*obj.n_Cineq);
                obj.delta_t_c = delta_z(3*obj.n_Cineq+1:end);
            elseif obj.Aflag == 1
                obj.delta_mu_l = delta_z(1:obj.n_Cineq);
                obj.delta_s_c = delta_z(obj.n_Cineq+1:2*obj.n_Cineq);
            elseif obj.Aflag == -1
                obj.delta_mu_u = delta_z(1:obj.n_Cineq);
                obj.delta_t_c = delta_z(obj.n_Cineq+1:2*obj.n_Cineq);
            end
            obj.delta_z = delta_z;
            obj.broadcastDeltaz();
        end

        function broadcastDeltaz(obj)
            % Send computed delta_mu back to all local nodes
            for i = 1:length(obj.nodes)
                obj.nodes{i}.receiveDeltaz(obj.delta_z);
            end
        end

        function obj = aggregateAndSolvestep(obj)
            % Initialize accumulators
            alphas = zeros(length(obj.nodes)+1,1);
            alphas(end) = min([1 ; (- 0.9 * obj.s_c) ./ obj.delta_s_c ; (- 0.9 * obj.t_c) ./ obj.delta_t_c]);
            f = 0;
            gradfT_deltax = 0;
            Gx = 0;
            Hxl = 0;
            Hxu = 0;
            Ax = zeros(obj.n_Cineq, 1);
            % Aggregate data from all local nodes
            for i = 1:length(obj.nodes)
                [fi, gradfT_deltaxi, Gxi, Hxli, Hxui, Axi, alphai] = obj.nodes{i}.sendDataToCentralstep(0);
                f = f + fi;
                gradfT_deltax = gradfT_deltax + gradfT_deltaxi;
                Gx = Gx + Gxi;
                Hxl = Hxl + Hxli;
                Hxu = Hxu + Hxui;
                alphas(i) = alphai;
                Ax = Ax + Axi;
%                 obj.Ax_sum = obj.Ax_sum + Ax_i;
%                 obj.Adelta_x_sum = obj.Adelta_x_sum + Adelta_x_i;
%                 obj.delta_taus(i) = delta_tau_i;
            end
            if obj.Aflag == 0
                Axr = norm(max(obj.lb - Ax, 0),1) + norm(max(Ax - obj.ub, 0),1);
            elseif obj.Aflag == 1
                Axr = norm(max(obj.lb - Ax, 0),1);
            elseif obj.Aflag == -1
                Axr = norm(max(Ax - obj.ub, 0),1);
            end
            alpha_ = min(alphas);
            obj.M1 = f + obj.sigma * (Gx + Hxl + Hxu + Axr);  
            obj.DM1 = gradfT_deltax - obj.sigma * (Gx + Hxl + Hxu + Axr);
%             gamma = 0.9;
            gamma = 1e-4;
            beta = 0.5;
            alpha_ = 1;
            Malphax = Malpha(obj, alpha_);
            while Malphax >= obj.M1 + gamma * alpha_ * obj.DM1
                alpha_ = alpha_ * beta;
                Malphax = Malpha(obj, alpha_);
            end
            obj.alpha = alpha_;
            for i = 1:length(obj.nodes)
                    obj.nodes{i}.receivealpha(obj.alpha);
            end
            if obj.Aflag == 0
                obj.mu_l = obj.mu_l + obj.alpha * obj.delta_mu_l;
                obj.mu_u = obj.mu_u + obj.alpha * obj.delta_mu_u;
                obj.s_c = obj.s_c + obj.alpha * obj.delta_s_c;
                obj.t_c = obj.t_c + obj.alpha * obj.delta_t_c;
            elseif obj.Aflag == 1
                obj.mu_l = obj.mu_l + obj.alpha * obj.delta_mu_l;
                obj.s_c = obj.s_c + obj.alpha * obj.delta_s_c;
            elseif obj.Aflag == -1
                obj.mu_u = obj.mu_u + obj.alpha * obj.delta_mu_u;
                obj.t_c = obj.t_c + obj.alpha * obj.delta_t_c;
            end
        end

        function Mxalpha = Malpha(obj, alpha)
                f = 0;
                Gx = 0;
                Hxl = 0;
                Hxu = 0;
                Ax = zeros(obj.n_Cineq, 1);
            for i = 1:length(obj.nodes)
                [fi, ~, Gxi, Hxli, Hxui, Axi, ~] = obj.nodes{i}.sendDataToCentralstep(alpha);
                f = f + fi;
                Gx = Gx + Gxi;
                Hxl = Hxl + Hxli;
                Hxu = Hxu + Hxui;
                Ax = Ax + Axi;
            end
            if obj.Aflag == 0
                Axr = norm(max(obj.lb - Ax, 0),1) + norm(max(Ax - obj.ub, 0),1);
            elseif obj.Aflag == 1
                Axr = norm(max(obj.lb - Ax, 0),1);
            elseif obj.Aflag == -1
                Axr = norm(max(Ax - obj.ub, 0),1);
            end
            Mxalpha = f + obj.sigma * (Gx + Hxl + Hxu + Axr);
        end
        
        function exit_flag = barrierstep(obj)
            Ax = zeros(obj.n_Cineq, 1);
            r = [0];
            for i = 1:length(obj.nodes)
                [ri, Axi] = obj.nodes{i}.sendresidual();
                Ax = Ax + Axi;
                r = [r; ri];
            end
            if obj.Aflag == 0                
                r1 = [obj.Ax_sum - obj.s_c - obj.lb;
                         obj.Ax_sum + obj.t_c - obj.ub];
            elseif obj.Aflag == 1
                r1 = [obj.Ax_sum - obj.s_c - obj.lb];
            elseif obj.Aflag == -1
                r1 = [obj.Ax_sum + obj.t_c - obj.ub];
            end
            if norm([r:r1],inf) < obj.tau
                obj.tau = max(0.1*obj.tau, obj.tol);
            end
            if obj.tau < obj.tol || norm([r:r1],inf) < obj.tol
                exit_flag = 1;
            end
        end

    end
end
