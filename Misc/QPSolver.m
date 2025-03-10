function [x,lambda,value] = QPSolver(H, f, A, lb, ub)
%% Solver for QP problem using the qpOASES idea
% Problem: min_x 1/2 x' H x + f' x
%          S.t.  lb <= A x <= ub       
% using a homotopy and active-set method inspired by qpOASES
%
% Inputs:
%   H:  (n x n) Symmetric positive definite matrix
%   f:  (n x 1) Gradient vector 
%   A:  (m x n) Constraint matrix
%   lb: (m x 1) lower bound 
%   ub: (m x 1) upper bound
%
% Outputs:
%   x - (n x 1) Vector of decision variables
%   lambda - (m x 1) Vector of dual variables
%   val - Scalar, optimal value
%

%% Initialization:
[m, n] = size(A);
x = zeros(n,1);
x0 = x;
% Compute unconstrained optimum: H*x* + f = 0  ==>  x* = -H\f
x_star = -H\f;
tau = 0;                    % Homotopy parameter
MaxIter = 3000;             % Maximum number of iterations of active set method
M = 1e+12;                  % In case there is no upper or lower bound, M or -M is substituted
H = H + 1e-12*eye(n);        % Small regulaization for numerical stability

% Tolerances to compare with zero
eps_den = 1e-15;
eps_num = 1e-15;
eps_LI = 1e-15;
eps_NZC = 1e-15;
tol = 1e-15;

% Perturbation parameters
r_initial = 0.5;
r_final = 1;
constraints = 1:m;
r_values = r_initial * (m - constraints) / (m - 1) + r_final * (constraints - 1) / (m - 1);
r = repmat(r_values, 3, 1);

% Assigning upper or lower bound in case it isnt provided
if isempty(lb)
    lb = -M * ones(m,1);
end
if isempty(ub)
    ub = M * ones(m,1);
end

temp = zeros(2);
temp2 = zeros(2);
lambda = zeros(m,1);

% Get initial working set via the crashing algorithm
% W = crashing(A, lb, ub, x0, x_star);
W = zeros(m,1);
W_new = W;
activeset = find(W ~= 0);
    
% Final homotopy data (tau = 1 data)
f_1 = f;
lb_1 = lb;
ub_1 = ub;
Delta1 = [f_1; lb_1; ub_1];

% Initial data (tau = 0) data
% f_0 = zeros(n,1);
lb_0 = zeros(m,1);
ub_0 = zeros(m,1);

% Perturb Lagrange multipliers
lambda(W == -1) = lambda(W == -1) + r(1, W == -1)'; % Lower bound active
lambda(W == +1) = lambda(W == +1) - r(1, W == +1)'; % Upper bound active
% Compute perturbed lower bounds lb(0)
lb_0(W ~= -1) = lb_0(W ~= -1) - r(2, W ~= -1)'; % Apply perturbation only at tau = 0
% Compute perturbed lower bounds ub(0)    
ub_0(W ~= 1) = ub_0(W ~= 1) + r(3, W ~= 1)';    % Apply perturbation only at tau = 0
% Compute adapted gradient f(0)
f_0 = A' * lambda - H * x;

    %
    %% Active set iteration
    
    for iter = 1:MaxIter
            
            % Extract rows of A corresponding to currently active constraints
            Aw = A(activeset,:);
            nA = length(activeset);
            % Initialize vector of bounds of active constrains
            aw_tau = zeros(nA,1);
            aw = zeros(nA,1);
            
            % Update homotopy data
            lb_tau = (1 - tau) * lb_0 + tau * lb_1;
            ub_tau = (1 - tau) * ub_0 + tau * ub_1;
            f_tau = (1 - tau) * f_0 + tau * f_1;

            DeltaTau = [f_tau; lb_tau; ub_tau];
%             nData = length(DeltaTau);
%             s = zeros(nData,1);

            denom = max(abs(DeltaTau), abs(Delta1)); % Element-wise max
            idx = denom < tol;  % Logical index where denom is smaller than tol
            
            s = (DeltaTau - Delta1) ./ denom; % Compute s element-wise
            s(idx) = 0;  % Set values where denom < tol to 0

            delta = norm(s, Inf);

            % Extract values of bounds for currently active upper and lower constraints
            if ~isempty(activeset)
                idx_neg = W(activeset) == -1;
                idx_pos = W(activeset) == 1;
                
                aw_tau(idx_neg) = lb_tau(activeset(idx_neg));
                aw(idx_neg) = lb(activeset(idx_neg));
                
                aw_tau(idx_pos) = ub_tau(activeset(idx_pos));
                aw(idx_pos) = ub(activeset(idx_pos));
            end

           
            % Check consistency of active econstraints
            res = Aw * x - aw_tau;
            mask = abs(res) > 1e-3;
            if ~all(mask == 0)
                [~,indtemp] = max(res(mask));
                inaind = activeset(indtemp);
                W_new(inaind) = 0;
                lambda(W_new == 0) = 0;
                W = W_new;
                activeset = find(W ~= 0);
                continue;
            end
            
            % Final target active bounds
            aw_1 = aw;

            % Build the KKT system for current tau
            %   [  H     Aw'   ]
            %   [  Aw     0    ]
            %
            KKT = [H,   Aw';...
                   Aw,  zeros(nA)];

            % Build the right-hand side
            %   [  - ( f((1) - f(tau) )  ]
            %   [    aw(1) - aw(tau)  ]
            %
            RHS = [-( f_1 - f_tau);...
                    aw_1 - aw_tau];


            % Solve for [delta_x; - delta_lambda]
            % Sol = KKT \ RHS;
            %--------------------------------------------------------------

            [L, D, P] = ldl(KKT);   % LDL^T factorization

            PRHS = P' * RHS;        % Compute P^T * RHS
            t = L \ PRHS;           % Forward solve
            t2 = D \ t;             % Diagonal solve
            z = L' \ t2;            % Backward solve
            
            DIR = P * z;            % Apply permutation to get the solution x

            %--------------------------------------------------------------

            DIR = DIR./(1-tau);
            % Extract direection in x (delta_x) 
            delta_x = DIR(1:n);

            % Extract Lagrange multipliers of currently active constraints
            delta_lambda_temp = - DIR(n+1:end);
            delta_lambda = zeros(m,1);
            % Insert delta_lambda of active constraints back into the full delta_lambda
            for j = 1:nA
                delta_lambda(activeset(j)) = delta_lambda_temp(j);
            end


            %---------------------------
            % STEP LENGTH (BLOCKING and SIGN CHANGE)
            %---------------------------
            % Among inactive constraints, see which one becomes active first

            InactiveInequalities = setdiff(1:m, activeset);
            A_inactive = A(InactiveInequalities,:);

            lb_inactive_tau = lb_tau(InactiveInequalities);
            ub_inactive_tau = ub_tau(InactiveInequalities);
            
            [temp(1,1), temp(1,2)] = RT( A_inactive*x - lb_inactive_tau, - A_inactive* delta_x, eps_den, eps_num, InactiveInequalities);
            [temp(2,1), temp(2,2)] = RT( ub_inactive_tau - A_inactive*x, A_inactive* delta_x, eps_den, eps_num, InactiveInequalities);

%             [temp(1,1), temp(1,2)] = RT( A_inactive*x - lb_inactive_tau, lb_1(InactiveInequalities) - lb_0(InactiveInequalities) - A_inactive* delta_x, eps_den, eps_num, InactiveInequalities);
%             [temp(2,1), temp(2,2)] = RT( ub_inactive_tau - A_inactive*x, A_inactive* delta_x + ub_0(InactiveInequalities) - ub_1(InactiveInequalities), eps_den, eps_num, InactiveInequalities);
 
            [delta_tau_p, idx] = min(temp(:,1));
            l = temp(idx,2);

            % Among active constraints, see which one is violated most
            [delta_tau_da, ka] =  RT( - W .* lambda, W .* delta_lambda, eps_den, eps_num, []);

            % If a constraint is weakly active, treat it like inactive
            WeakIdx = find(W~= 0 & abs(lambda) < 1e-15);
            A_weak = A(WeakIdx,:);
            lb_weak_tau = lb_tau(WeakIdx);
            ub_weak_tau = ub_tau(WeakIdx);
            [temp2(1,1), temp2(1,2)] = RT( A_weak*x - lb_weak_tau, - A_weak* delta_x, eps_den, eps_num, WeakIdx);
            [temp2(2,1), temp2(2,2)] = RT( ub_weak_tau - A_weak*x, A_weak* delta_x, eps_den, eps_num, WeakIdx);
            
            [delta_tau_w, idx2] = min(temp2(:,1));
            lw = temp2(idx2,2);
            
            % Choose smallest valid step
            delta_tau = min([delta_tau_p, delta_tau_da,delta_tau_w]);
            
            % Update tau, x and lambda
            if (delta_tau >= 1 - tau) || delta_tau == inf
                x = x + (1 - tau) * delta_x;
                lambda = lambda + (1 - tau) * delta_lambda;
                tau = 1;
                break;
            else
                tau = tau + delta_tau;
                x = x + delta_tau * delta_x;
                lambda = lambda + delta_tau * delta_lambda;
                %----------------------------------------------------------
                if delta <= 1e-15
                    break;
                end
                %----------------------------------------------------------
            end

            % Add to working set in case of primal blocking
            if delta_tau == delta_tau_p
                  if idx == 1
                      W_new(l) = -1;
                  elseif idx == 2
                      W_new(l) = 1;
                  end
            end

            %--------------------------------------------------------------
            % Determining linear dependence and an exchange constraint to... 
            % modify active and working set

            % Solve for p and q_w in 
            %   [  H     Aw'   ] [  p  ] = [ A_l^{T} ]
            %   [  Aw     0    ] [ q_w ] = [    0    ]
            %  A_l linearly dependent on A_w if and only if p = 0
            if ~isnan(l)

                PA_l = P' * [A(l,:)'; zeros(nA,1)];        % Compute P^T * RHS
                u = L \ PA_l;           % Forward solve
                u2 = D \ u;             % Diagonal solve
                u3 = L' \ u2;           % Backward solve
                pq_w = P * u3;          % Apply permutation to get the solution [p; q_w]

                p = pq_w(1:n);
                q_w = pq_w(n+1:end);
                q = zeros(m,1);
                q(activeset) = q_w;
                zeta_q = [p; q];
                if norm(p,inf) <= eps_LI * norm(zeta_q,inf)
                    [psi_star, kappa] = RT(- W_new(l) * (W .* lambda), W_new(l)* (W .* q), eps_den, eps_num, []);
                    if ~isnan(kappa)
                        W_new(kappa) = 0;
                        lambda = lambda + psi_star * W_new .* q;
                        lambda(l) = - psi_star * W_new(l);
                    end
                end
            end
            %--------------------------------------------------------------

            % Subtract from working set in case of dual blocking
            if delta_tau == delta_tau_da
                if ~isnan(ka)
                    W_new(ka) = 0;
                end
            end
            
            %--------------------------------------------------------------
            % Determining zero curvature and an exchange constraint to... 
            % modify active and working set
            
            % Solve for p and q_w in 
            %   [  H     Aw'   ] [   s  ] = [     0     ]
            %   [  Aw     0    ] [ xi_w ] = [ - (e_k)_w ]
            %  H singular on the null space of A_w_new if and only if xi_w = 0

            if ~isnan(ka)
                e_k = zeros(m,1);
                e_k(ka) = 1;
                e_k_w = e_k(W~=0);
                rhs = [zeros(n,1); - e_k_w];
                Prhs = P' * rhs;          % Compute P^T * b
                y = L \ Prhs;             % Forward solve
                y2 = D \ y;               % Diagonal solve
                y3 = L' \ y2;             % Backward solve
                dir = P * y3;             % Apply permutation to get the solution x

                s = dir(1:n);
                xi_w = dir(n+1:end);
                zeta_s = [s; xi_w];
                if norm(xi_w,inf) <= eps_NZC * norm(zeta_s,inf)
                    [sigma_l, elle1] = RT(A * x - lb, - A * s, eps_den, eps_num, []);
                    [sigma_u, elle2] = RT(ub - A * x, A * s, eps_den, eps_num, []);
                    sigma = min(sigma_l, sigma_u);
                    if sigma == sigma_l
                        W_new(elle1) = -1;
                    elseif sigma == sigma_u
                        W_new(elle2) = 1;
                    end
                    x = x + sigma * s;  
                end
            end
            %--------------------------------------------------------------

            W = W_new;
            activeset = find(W ~= 0);

    end
    value = 1/2 * x' * H * x + f' * x;

    %----------------------------------------------------------------------
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
% %         if isempty(w)
% %             index = ind;
% %         else
%             index = validIndices(ind);
% %         end
    end

    %----------------------------------------------------------------------
    function W = crashing(A, lb, ub, x0, x_star, tol)
        % CRASHINGALGORITHM selects an initial working set for a QP solver.
        %
        %   W = crashing(A, lb, ub, x0, tol) computes a vector W
        %   indicating which constraints are active (or should be initially 
        %   treated as active) at the starting point x0.
        %
        %   Input:
        %     A   - m-by-n constraint matrix (for inequalities lb <= A*x <= ub)
        %     lb  - m-by-1 vector of lower bounds
        %     ub  - m-by-1 vector of upper bounds
        %     x0  - n-by-1 initial point (often x0 = zeros(n,1))
        %     tol - tolerance for considering a constraint violated (default: 1e-6)
        %
        %   Output:
        %     W   - m-by-1 working set indicator vector:
        %           W(i) = -1 means constraint i is active at its lower bound,
        %           W(i) =  1 means constraint i is active at its upper bound,
        %           W(i) =  0 means constraint i is inactive.
        %
        % This simple crashing algorithm examines each constraint's violation
        % at the initial point x0, sorts them in order of most to least violated,
        % and then selects those whose violation exceeds a tolerance.
        
        if nargin < 6
            tol = 1e-6;
        end
        
        [m, n] = size(A);
        W = zeros(m,1);
        
        % Compute A*x0 once for all constraints.
        Ax0 = A * x0;

        % Compute violation for each constraint
        % For constraint i:
        %   v_low = max(lb(i) - A(i,:)*x0, 0)   (violation of lower bound)
        %   v_up  = max(A(i,:)*x0 - ub(i), 0)   (violation of upper bound)
        % Total violation is the sum.
%         violations = zeros(m,1);
        v_low = max(lb - Ax0, 0);
        v_up  = max(Ax0 - ub, 0);
        violations = v_low + v_up;
        
        % Sort constraints by violation magnitude (largest first)
        [~, idx_sort] = sort(violations, 'descend');
        
        % Add constraints with violation above tol to the working set.
        % For each constraint, decide whether it is more violated from below (activate as lower bound)
        % or from above (activate as upper bound).
    
        if all(violations == 0)
            % Compute the direction from x0 to x_star
            d = x_star - x0;

%             % Compute A*d for all constraints at once.
%             dval = A * d;
            % For each constraint, compute the step length (alpha) required so that
            % x0 + alpha*d exactly reaches the boundary.
            % For constraint i: A(i,:)*x must equal either lb(i) or ub(i)

            alphas = Inf(m,1);
            for il = 1:m
                ai = A(il,:);
                val0 = ai*x0;
                dval = ai*d;
                % If dval > 0 then moving along d increases ai*x,
                % so the upper bound will be hit: alpha = (ub(i)-val0)/dval.
                if dval > 0
                    alphas(il) = (ub(il)-val0)/dval;
                % If dval < 0 then moving along d decreases ai*x,
                % so the lower bound will be hit: alpha = (lb(i)-val0)/dval.
                elseif dval < 0
                    alphas(il) = (lb(il)-val0)/dval;
                end
                % If dval == 0, this constraint is not affected by motion in d.
            end
            
            % We need the smallest positive step (alpha) among all constraints.
            posAlphas = alphas(alphas > 0);
            if isempty(posAlphas)
                % If no constraint would ever be hit, return an empty working set.
                W = zeros(m,1);
                return;
            end
            [min_alpha, ~] = min(posAlphas);
            
            % Compute the "crashed" point: x_crash = x0 + min_alpha*d
            x_crash = x0 + min_alpha*d;
            
            % Now, for each constraint, decide if it is active at x_crash.
            % That is, check if A(i,:)*x_crash is (within tolerance) equal to lb(i) or ub(i).
            W = zeros(m,1);
            for i = 1:m
                ai = A(i,:);
                val = ai*x_crash;
                if abs(val - lb(i)) < tol
                    W(i) = -1;  % Lower bound active.
                elseif abs(val - ub(i)) < tol
                    W(i) = 1;   % Upper bound active.
                end
            end
        else
            for k = 1:m
                i = idx_sort(k);
                if violations(i) > tol
                    Ax = A(i,:) * x0;
                    if Ax < lb(i)
                        W(i) = -1;  % Activate as lower-bound constraint.
                    elseif Ax > ub(i)
                        W(i) = 1;   % Activate as upper-bound constraint.
                    end
                else
                    % Once violations fall below tolerance, stop.
                    break;
                end
            end
        
        end
    end
    %----------------------------------------------------------------------

end
