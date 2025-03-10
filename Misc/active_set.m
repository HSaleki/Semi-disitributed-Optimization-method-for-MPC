function [x, lambda, mu, val] = active_set(B, f, G, g, H, h)

    %% A function for solving QP using active set methods
    % Problem: min 1/2 x'Bx + f'x
    %          s.t. G'x + g = 0         (equality constraints)
    %               H'x + h <= 0        (inequality constraints)
    %% Inputs:
    %   B: (n x n) Symmetric positive definite matrix
    %   f: (n x 1) Gradient vector 
    %   G: (n x nG) Equality constraint matrix
    %   g: (nG x 1) Equality constraint bound
    %   H: (n x nH) Inequality constraint matrrix
    %   h: (nH x 1) Inequality constraint bound
    %
    %% Outputs:
    %   x - (n x 1) Vector of decision variables
    %   lambda - (nG x n)  Vector of equality dual variables
    %   mu - (nH x 1) Vector of inequality dual variables
    %   val - Scalar, optimal value
    %
    %% Initialization:
    n = size(B,1);
    nG = size(G,2);
    ng = length(g);
    nH = size(H,2);
    nh = length(h);

    tol = 1e-12;
    MaxIter = 300;

    if nG ~= ng
        disp('Equality constraint dimnsion mismatch')
        return
    elseif nH ~= nh
        disp('Inequality constraint dimnsion mismatch')
        return
    end

%---------------------------------------------------------------
    % Needs a phase I here to find an initial feasible point
    x = zeros(n,1);
%---------------------------------------------------------------

    lambda = zeros(nG,1);
    mu = zeros(nH,1);
    activeset = [];

    %% Active set iteration
    for i = 1:MaxIter
        % Re-initialize mu at each iteration
        mu = zeros(nH,1);  

        % Extract columns of H corresponding to currently active constraints
        H_A = H(:,activeset);

        % Build the KKT system
        %   [  B     G     H_A    ]
        %   [  G'    0      0     ]
        %   [ H_A'   0      0     ]
        %
        KKT = [B,      G,                             H_A;...
               G',     zeros(nG),                     zeros(nG, length(activeset));...
               H_A',   zeros(length(activeset), nG),  zeros(length(activeset))];

        % Regularize for numerical stability
        %KKT = KKT + 1e-6 * eye(length(KKT));

        % Build the right-hand side
        %   [  B*x + f  ]
        %   [     0     ]
        %   [     0     ]
        %
        RHS = - [B * x + f;...
                 zeros(nG,1);...
                 zeros(length(activeset),1)];

        % Solve for [d; lambda; mu_A]
        Sol = KKT \ RHS;

        % Extract the step direction d
        d = Sol(1:n);

        % Extract equality multipliers (lambda) and active inequality multipliers (mu_A)
        if length(Sol) > n
            if nG > 0
                lambda = Sol(n+1 : n+nG);
            end
            mu_A = Sol(n+nG+1 : end);
        else
            mu_A = [];
        end
        
        % Insert mu_A back into the full mu
        for j = 1:length(activeset)
            mu(activeset(j)) = mu_A(j);
        end

        %---------------------------
        % KKT COMPLEMENTARY CHECK
        %---------------------------
        % If the direction is small enough, we check Lagrange multipliers
        if norm(d) <= tol
            % If all active-set multipliers are >= 0, or if no active constraints
            if all(mu >= 0) || isempty(activeset)
                % We have a KKT point
                break;
            end
            % Otherwise remove the constraint with the most negative mu
            [~, ind] = min(mu);
            activeset(activeset == ind) = [];
        end
        
        %---------------------------
        % STEP LENGTH (BLOCKING)
        %---------------------------
        % Among inactive constraints, see which one becomes active first
        InactiveInequalities = setdiff(1:nH, activeset);

        temp = zeros((nH - length(activeset)), 2);
        for j = 1:length(InactiveInequalities)
            idx = InactiveInequalities(j);
            denom = H(:,idx)' * d;
            if denom > 0
                % alpha = fraction to make H(:,idx)'*(x + alpha*d) + h(idx) = 0
                % => alpha = [ -h(idx) - H(:,idx)'*x ] / [H(:,idx)'*d]
                num = -(h(idx) + H(:,idx)'*x);
                temp(j,:) = [ num / denom, idx ];
            else
                % If denom <= 0, that constraint is not 'blocking'
                temp(j,:) = [ 1e12, idx ];
            end
        end

        [alpha_min, row_min] = min(temp(:,1));
        new_constraint_idx = temp(row_min, 2);
        if alpha_min < 1
            alpha = alpha_min;
            x = x + alpha * d;
            activeset = [activeset; new_constraint_idx];
            activeset = unique(activeset);
        else
            %alpha = 1;
            x = x + d;
            % No new constraint is added
        end

        % Compute the objective value
        val = 0.5 * x' * B * x + f' * x;
    end




    


