classdef CentralNode < handle
    properties
        nodes           % Array of LocalNode instances
        mu              % Global Lagrange multiplier
        delta_mu        % Global Lagrange multiplier change
        lb              % Global constraint lower bound vector
        ub              % Global constraint upper bound vector
        Wc              % Working set (W_i = -1 for lower active, 0 inactive, +1 for upper active constraints)
        activeset_c     % Active set for coupled inequality constraint
        n               % Number of inequality constraints (size(A,1))
        lb_tau          % Lower bound at tau
        ub_tau          % Upper bound at tau
        lb_0            % Initial lower bound
        ub_0            % Initial upper bound
        aw              % Active inequalities bound vector
        aw_0            % Initial active inequalities bound vector
        aw_tau          % Active inequalities bound vector at tau
        Delta1          % Vector of final gradient and bounds for stopping criterion
        DeltaTau        % Vector of gradient and bounds at tau for stopping criterion
        r               % Perturbation matrix
        tau             % Homotopy parameter
        delta_tau       % Homotopy parameter step
        Ax_sum          % Sum_i A_i * x_i   
        Adelta_x_sum    % Sum_i A_i * delta_x_i
        delta_taus      % Vector of delta_tau from all local nodes and central one
        stopflag        % Flag for stopping when reached the accuracy
        flagc           % Flag for reaching coupled constrained accuracy
        exit_flag       % Exit flag for main script
    end
    
    methods
        function obj = CentralNode(nodes, lb, ub)
            obj.nodes = nodes;
            obj.lb = assignIfEmpty(lb, size(ub), -1);
            obj.ub = assignIfEmpty(ub, size(lb), 1);
            obj.n = max(length(obj.lb), length(obj.ub));
            obj.mu = zeros(obj.n,1);        % Initialize inequality multipliers
            obj.Wc = zeros(obj.n,1);        % Initialize will all inequalities inactive
            obj.delta_mu = zeros(obj.n,1);  % % Initialize inequality multipliers change
            obj.activeset_c = find(obj.Wc ~= 0);
            obj.tau = 0;
            obj.delta_tau = 0;
            obj.exit_flag = 0;
            obj.delta_taus = zeros(length(obj.nodes) + 1,1);
            obj.stopflag = zeros(length(obj.nodes)+1,1);
            obj.Initialize();
            obj.updatetauvalues();
            obj.flagc = 0;
        end

    function obj = Initialize(obj)
            obj.lb_0 = zeros(obj.n,1);
            obj.ub_0 = zeros(obj.n,1);
            obj.Delta1 = [obj.lb; obj.ub];
            constraints = 1:obj.n;
            if obj.n > 1
                r_initial = 0.5;
                r_final = 1;
                r_values = r_initial * (obj.n - constraints) / (obj.n - 1) + r_final * (constraints - 1) / (obj.n - 1);
                obj.r = repmat(r_values, 3, 1);
                % Perturb Lagrange multipliers
                obj.mu(obj.Wc == -1) = obj.mu(obj.Wc == -1) + obj.r(1, obj.Wc == -1)'; % Lower bound active
                obj.mu(obj.Wc == +1) = obj.mu(obj.Wc == +1) - obj.r(1, obj.Wc == +1)'; % Upper bound active
                % Compute perturbed lower bounds lb(0)
                obj.lb_0(obj.Wc ~= -1) = obj.lb_0(obj.Wc ~= -1) - obj.r(2, obj.Wc ~= -1)'; % Apply perturbation only at tau = 0
                % Compute perturbed lower bounds ub(0)    
                obj.ub_0(obj.Wc ~= 1) = obj.ub_0(obj.Wc ~= 1) + obj.r(3, obj.Wc ~= 1)';    % Apply perturbation only at tau = 0
            end
        end
        
        function obj = updatetauvalues(obj)
%             obj.tau = obj.tau + obj.delta_tau;
            % Compute adaptive bounds and gradient
            obj.lb_tau = (1 - obj.tau) * obj.lb_0 + obj.tau * obj.lb;
            obj.ub_tau = (1 - obj.tau) * obj.ub_0 + obj.tau * obj.ub;
            obj.DeltaTau = [obj.lb_tau; obj.ub_tau];
            denom = max(abs(obj.DeltaTau), abs(obj.Delta1)); % Element-wise max
            idx = denom < 1e-15;  % Logical index where denom is smaller than tol
            s = (obj.DeltaTau - obj.Delta1) ./ denom; % Compute s element-wise
            s(idx) = 0;  % Set values where denom < tol to 0             
            delta = norm(s, Inf);
            if delta <= 1e-15
                obj.flagc = 1;
            end
        end

        function delta_mu = aggregateAndSolveDeltaMu(obj)
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
            
            % Solve for delta_mu: (sum_i M_i) * delta_mu = r - (sum_i v_i)
            delta_mu = M_sum \ (obj.aw - obj.aw_tau - v_sum);
            delta_mu = delta_mu / (1 - obj.tau);
            obj.delta_mu(obj.activeset_c) = delta_mu; 
        end

        function broadcastDeltaMu(obj)
            % Send computed delta_mu back to all local nodes
            for i = 1:length(obj.nodes)
                obj.nodes{i}.receiveDeltaMu(obj.delta_mu);
            end
        end

        function obj = aggregateAndSolvestep(obj)
            % Initialize accumulators
            n_inactive = obj.n - length(obj.activeset_c);
            obj.Ax_sum = zeros(n_inactive,1);
            obj.Adelta_x_sum = zeros(n_inactive,1);
                        
            % Aggregate data from all local nodes
            for i = 1:length(obj.nodes)
                [Ax_i, Adelta_x_i, delta_tau_i] = obj.nodes{i}.sendDataToCentralstep();
                obj.Ax_sum = obj.Ax_sum + Ax_i;
                obj.Adelta_x_sum = obj.Adelta_x_sum + Adelta_x_i;
                obj.delta_taus(i) = delta_tau_i;
            end
                                  
        end

        function exit_flag = homotopystep(obj)

            InactiveInequalities = setdiff(1:obj.n, obj.activeset_c);
            lb_inactive_tau = obj.lb_tau(InactiveInequalities);
            ub_inactive_tau = obj.ub_tau(InactiveInequalities);
            
            [temp(1,1), temp(1,2)] = RT( obj.Ax_sum - lb_inactive_tau, obj.lb(InactiveInequalities) - obj.lb_0(InactiveInequalities) - obj.Adelta_x_sum, 1e-15, 1e-15, InactiveInequalities);
            [temp(2,1), temp(2,2)] = RT( ub_inactive_tau - obj.Ax_sum, obj.Adelta_x_sum + obj.ub_0(InactiveInequalities) - obj.ub(InactiveInequalities), 1e-15, 1e-15, InactiveInequalities);
 
            [delta_tau_p, idx] = min(temp(:,1));
            l = temp(idx,2);

            % Among active constraints, see which one is violated most
            [delta_tau_da, ka] =  RT( - obj.Wc .* obj.mu, obj.Wc .* obj.delta_mu, 1e-15, 1e-15, []);

            % Choose smallest valid step
            obj.delta_tau = min([delta_tau_p, delta_tau_da]);
            obj.delta_taus(end) = obj.delta_tau;

            [delta_tau, id] = min(obj.delta_taus);
            obj.delta_tau = delta_tau; 
            if (obj.delta_tau >= 1 - obj.tau) || obj.delta_tau == inf
                obj.exit_flag = 1;
            end
            if id == length(obj.delta_taus)
                % Coupled constrained limits step length
                if (obj.delta_tau >= 1 - obj.tau) || obj.delta_tau == inf
                    obj.mu = obj.mu + (1 - obj.tau) * obj.delta_mu;
                    obj.tau = 1;
                    obj.exit_flag = 1;
                else
                    obj.tau = obj.tau + obj.delta_tau;
                    obj.mu = obj.mu + obj.delta_tau * obj.delta_mu;
                    %----------------------------------------------------------
                    if all(obj.stopflag == 1)
                        obj.exit_flag = 1;
                    end
                    %----------------------------------------------------------
                end
                % Add to working set in case of primal blocking
                if obj.delta_tau == delta_tau_p
                      if idx == 1
                          W_new(l) = -1;
                      elseif idx == 2
                          W_new(l) = 1;
                      end
                elseif obj.delta_tau == delta_tau_da
                    if ~isnan(ka)
                        W_new(ka) = 0;
                    end
                end
                obj.Wc = W_new;
                obj.activeset_c = find(obj.Wc ~= 0);
            else
                for i = 1:length(obj.nodes)
                    if i == id
                        obj.nodes{i}.receiveDeltatau(obj.delta_tau, 1, obj.Wc);
                    else
                        obj.nodes{i}.receiveDeltatau(obj.delta_tau, 0, obj.Wc);
                    end
                end
            end
            obj.tau = obj.tau + obj.delta_tau;
            exit_flag = obj.exit_flag;
        end

    end
end

function [val, index] = RT(u, v, eps_den, eps_num, w)

        % A function to perform ratio test for finding step lenth
        % Inputs:
        %   u - vector of numerators
        %   v - vector of denominators
        %   w - vector of inactive inequalities
        % Output:
        % [val, index] = RT(u,v) = min_i{u_i/v_i | 1 <= i <= m, v_i > 0}

        u_cut = max(u, 0);

        Mask = (v >= eps_den) & (u_cut >= eps_num);

        if ~any(Mask)
            % No valid constraints
            val    = Inf;
            index = NaN;
            return;
        end
        
        if isempty(w)
            vec = (1:length(u))';
            validIndices = vec(Mask); 
            validRatios = u_cut(Mask) ./ v(Mask);
            [val, ind] = min(validRatios);
            index = validIndices(ind);
        else
            validIndices = w(Mask);
            validRatios = u_cut(Mask) ./ v(Mask);
            [val, ind] = min(validRatios);
            index = validIndices(ind);
        end
end


function val = assignIfEmpty(input, n, flag)
    if ~isempty(input)
        val = input;
    else
        val = flag*ones(n)*1e8;
    end
end