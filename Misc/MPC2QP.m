function [H,f,A,b,Aeq,beq,lb,ub] = MPC2QP(Q, R, P, A, B, N, x_min, x_max, u_min, u_max, x_k)

    
    %% Linear Quadratic MPC Transformation to Quadratic Optimization Problem
    %   min J = sum_(i=0)^(i=N-1) (x_i' Q x_i + u_i' R u_i) + x_N' P x_N =
    %   X'Q_extended X + U' R_extended U + x_0' Q x_0
    %   S.t. x_k+1 = A x_k + B u_k
    %   x_min <= x_k <= x_max
    %   u_min <= u_k <= u_max
    %   Q_extended = blockdiag(Q,...,Q,P)
    %   R_extended = blobkdiag(R)
    %
    %% Inputs:
    %   A: state transition matrix (n x n)
    %   B: Input matrix (n x m)
    %   Q: State weighting matrix (n x n) 
    %   R: Input weighting matrix (m x m)
    %   P: Terminal state weighting matrix (n x n)
    %   N: Perdiction horizon (positive integer)
    %   x_k: current state vector (n x 1)
    %
    %% Outputs:
    %   F: Extended state transition matrix
    %   G: Extended input matrix
    %   H: Hessian matrix for the quadratic optimization problem
    %   f: Gradient vector for the quadratic optimization problem
    %
    %% Turned into:
    %               min_U J = U' H U + 2 f' U
    %               S.t. X = F x_k + G U
    %                or  G U = X - F x_k
    %                    X_min <= X <= X_max
    %                    U_min <= U <= U_max
    %   With X = [x_k+1; ...; x_k+N]
    %   U = [u_k; ...; u_k+N-1]
    %   F = [A; A^2; ...; A^N]
    %   G = [B     0 ... 0;...
    %        AB    B ... 0;...
    %        A^2B AB ... 0;...
    %        A^N-1B  ... B]
    %   H = G' Q_extended G + R_extended
    %   f = G' Q_extended F x_k 
    
    %% Input validation
    n = size(A, 1);
    m = size(B, 2);

    if N <= 0 || mod(N, 1) ~= 0
        error('Prediction horizon N must be a positive integer.');
    end

    if size(A, 1) ~= size(A, 2)
        error('Matrix A must be square.');
    end

    if size(B, 1) ~= n
        error('Matrix B must have the same number of rows as A.');
    end

    if size(Q, 1) ~= n || size(Q, 2) ~= n
        error('Matrix Q must be of size n x n.');
    end

    if size(R, 1) ~= m || size(R, 2) ~= m
        error('Matrix R must be of size m x m.');
    end

    if size(P, 1) ~= n || size(P, 2) ~= n
        error('Matrix P must be of size n x n.');
    end

    if size(x_k, 1) ~= n || size(x_k, 2) ~= 1
        error('State vector x_k must be of size n x 1.');
    end

    % Preallocate matrices for efficiency
    F = zeros(N * n, n);
    G = zeros(N * n, N * m);

    % Construct F and G
    A_power = eye(n);
    for i = 1:N
        A_power = A_power * A;
        F((i - 1) * n + 1:i * n, :) = A_power;

        % For G each clock in (i,j) should be A^(i-j)B
        % For i = j (diagonal) just put B
        % For j < i multiply last iteration by A
        for j = 1:i
            row_idx = (i - 1) * n + 1 : i * n;
            col_idx = (j - 1) * m + 1 : j * m;
            if j == i
                % Directly assign B for diagonal
                G(row_idx, col_idx) = B;
            else
                % Multiply last row blocks by A
                prev_row_idx = (i - 2) * n + 1 : (i - 1) * n;
                G(row_idx, col_idx) = A * G(prev_row_idx, col_idx);
            end
        end
    end

    % Construct Q_extended and R_extended
    Q_cells = repmat({Q}, 1, N - 1);
    Q_cells{N} = P;
    Q_extended = blkdiag(Q_cells{:});

    R_extended = kron(eye(N), R);

    % Compute H and f
    H = G' * Q_extended * G + R_extended;
    f = G' * Q_extended * F * x_k;
    
    % Assign outputs in quadprog format
    X_min = repmat(x_min,N,1);
    X_max = repmat(x_max,N,1);
    U_min = repmat(u_min,N,1);
    U_max = repmat(u_max,N,1);
    Aeq = [];
    beq = [];
    A = [G; -G];
    b = [X_max - F * x_k; - (X_min - F * x_k)];
    lb = U_min;
    ub = U_max;

end  % End of function
    