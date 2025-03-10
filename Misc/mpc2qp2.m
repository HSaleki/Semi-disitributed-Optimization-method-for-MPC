function [H, f, Aeq, beq, Aineq, bineq, lb, ub] = mpc2qp2( ...
    A, B, c, Q, R, P, N, ...
    v0, vRef, uRef, ...
    Ck, Dk, dlk, duk, varargin )
% mpc2qp  Converts a linear MPC problem with reference tracking into a QP.
%
% SYNTAX:
%   [H, f, Aeq, beq, Aineq, bineq, lb, ub] = mpc2qp( ...
%       A, B, c, Q, R, P, N, ...
%       v0, vRef, uRef, ...
%       Ck, Dk, dlk, duk )
%
%   ... = mpc2qp(..., 'TimeVaryingAB', true, 'Acell', ACell, 'Bcell', BCell, 'ccell', cCell)
%
% DESCRIPTION:
%   This function formulates the following linear MPC with reference tracking:
%
%       min   0.5*sum_{k=0}^{N-1} [ ||v_k - v_k^r||_{Q_k}^2 + ||u_k - u_k^r||_{R_k}^2 ]
%             + 0.5*||v_N - v_N^r||_P^2
%
%       subject to:
%           v_{k+1} = A_k*v_k + B_k*u_k + c_k,    k = 0,...,N-1
%           v_0 = v0
%           d_k^l <= C_k*v_k + D_k*u_k <= d_k^u,    k = 0,...,N-1
%           d_N^l <= C_N*v_N <= d_N^u
%
%   It returns the QP in standard form:
%
%       min_U   0.5*U'*H*U + f'*U
%        s.t.   Aeq*U = beq
%               Aineq*U <= bineq
%               lb <= U <= ub
%
%   where U = [u_0; u_1; ...; u_{N-1}] is the stacked control sequence.
%
% INPUTS:
%   A, B, c   : system matrices (time invariant by default).
%               A is n_v x n_v, B is n_v x n_u, c is n_v x 1.
%   Q, R      : weighting matrices for states & inputs.
%               Either as single matrices (time invariant) or as cell arrays {Q0,...,Q_{N-1}} and {R0,...,R_{N-1}}.
%   P         : terminal cost weighting (n_v x n_v).
%   N         : prediction horizon (integer).
%   v0        : initial state (n_v x 1).
%   vRef      : stacked reference states [v_1^r; ...; v_N^r] (N*n_v x 1).
%   uRef      : stacked reference inputs [u_0^r; ...; u_{N-1}^r] (N*n_u x 1).
%
%   Ck, Dk    : constraint matrices. They can be provided as either cell arrays of
%               length N+1 or as single matrices (which are then replicated for k=0,...,N).
%   dlk, duk  : lower and upper bound vectors for the constraints. Again, they can be provided
%               as cell arrays of length N+1 or as single arrays.
%
% OPTIONAL NAME-VALUE PAIRS:
%   'TimeVaryingAB' (default = false):
%       If true, you must pass 'Acell', 'Bcell', 'ccell' as cell arrays of length N.
%
% OUTPUTS:
%   H, f      : QP cost matrices so that cost = 0.5*U'*H*U + f'*U.
%   Aeq, beq  : QP equality constraints (usually empty if dynamics are incorporated).
%   Aineq, bineq : QP inequality constraints.
%   lb, ub    : Lower and upper bounds on U (if applicable, else empty).

%% Parse dimensions
n_v = size(A,1);  % state dimension
m_u = size(B,2);  % input dimension

%% Process Q and R weights
if ~iscell(Q)
    Qk = cell(1, N);
    for k = 1:N
        Qk{k} = Q;
    end
else
    if length(Q) ~= N
        error('Number of Q cell entries must match horizon N.');
    end
    Qk = Q;
end

if ~iscell(R)
    Rk = cell(1, N);
    for k = 1:N
        Rk{k} = R;
    end
else
    if length(R) ~= N
        error('Number of R cell entries must match horizon N.');
    end
    Rk = R;
end

%% Process constraint inputs: replicate if not provided as cell arrays
if ~iscell(Ck)
    temp = Ck;
    Ck = cell(1, N+1);
    for k = 1:(N+1)
        Ck{k} = temp;
    end
end

if ~iscell(Dk)
    temp = Dk;
    Dk = cell(1, N+1);
    for k = 1:(N+1)
        Dk{k} = temp;
    end
end

if ~iscell(dlk)
    temp = dlk;
    dlk = cell(1, N+1);
    for k = 1:(N+1)
        dlk{k} = temp;
    end
end

if ~iscell(duk)
    temp = duk;
    duk = cell(1, N+1);
    for k = 1:(N+1)
        duk{k} = temp;
    end
end

%% Check if time-varying A, B, c are provided
p = inputParser;
addParameter(p, 'TimeVaryingAB', false, @islogical);
addParameter(p, 'Acell', [], @(x) iscell(x) || isempty(x));
addParameter(p, 'Bcell', [], @(x) iscell(x) || isempty(x));
addParameter(p, 'ccell', [], @(x) iscell(x) || isempty(x));
parse(p, varargin{:});

timeVaryingAB = p.Results.TimeVaryingAB;
Acell = p.Results.Acell;
Bcell = p.Results.Bcell;
ccell = p.Results.ccell;

%% Build extended dynamics: V = F*v0 + G*U + w
% Decision variable U = [u_0; u_1; ...; u_{N-1}], with U in R^(N*m_u)
F = zeros(N*n_v, n_v);
G = zeros(N*n_v, N*m_u);
w = zeros(N*n_v, 1);

A_power = eye(n_v);  % used for time-invariant case
for i = 1:N
    idx_v = (i-1)*n_v + 1 : i*n_v;
    
    % Choose appropriate matrices for step i
    if timeVaryingAB
        Ai = Acell{i};
        Bi = Bcell{i};
        ci = ccell{i};
    else
        Ai = A;
        Bi = B;
        ci = c;
    end
    
    if timeVaryingAB
        F(idx_v,:) = buildFrowTimeVary(Acell, i, n_v);
    else
        A_power = A_power * Ai;
        F(idx_v,:) = A_power;
    end
    
    % Build offset vector w.
    if timeVaryingAB
        w(idx_v) = buildOffsetTimeVary(Acell, Bcell, ccell, v0, i, true);
    else
        temp = zeros(n_v,1);
        for j = 1:i
            temp = temp + A^(i-j)*c;
        end
        w(idx_v) = temp;
    end
    
    % Build G: each block corresponds to the influence of u_j on v_i
    for j = 1:i
        row_idx = (i-1)*n_v + 1 : i*n_v;
        col_idx = (j-1)*m_u + 1 : j*m_u;
        if timeVaryingAB
            G(row_idx, col_idx) = buildGblockTimeVary(Acell, Bcell, j, i);
        else
            G(row_idx, col_idx) = A^(i-1-j) * B;
        end
    end
end

%% Build extended weighting matrices Qext and Rext
% Qext is blkdiag(Q0, Q1, ..., Q_{N-1}, P)
Qcells = cell(1, N);
for k = 1:N-1
    Qcells{k} = Qk{k};
end
Qcells{N} = P;
Qext = blkdiag(Qcells{:});

Rcells = cell(1, N);
for k = 1:N
    Rcells{k} = Rk{k};
end
Rext = blkdiag(Rcells{:});

%% Build the stacked state vector and compute the offset for the cost
% Predicted states: V = F*v0 + G*U + w, with vRef stacked as [v1^r; ...; vN^r]
barF = (F*v0 + w) - vRef;
if ~isempty(uRef)
    if size(uRef,1) ~= N*m_u
        error('uRef must have length N*m_u.');
    end
    barF = barF + G*uRef;
end

%% Construct QP cost: H and f
H = G' * Qext * G + Rext;
f = G' * Qext * barF;

%% Equality constraints
% Dynamics have been folded into F, G, and w, so equality constraints are empty.
Aeq = [];
beq = [];

%% Build inequality constraints from:
%   d_k^l <= C_k*v_k + D_k*u_k <= d_k^u, for k = 0,...,N.
% Here, v_k is known for k = 0 (v0) and predicted for k>=1.
numConstraints = 0;
for k = 0:N
    nC = size(Ck{k+1},1);
    numConstraints = numConstraints + 2*nC;
end

Aineq = zeros(numConstraints, N*m_u);
bineq = zeros(numConstraints, 1);
rowStart = 1;
for k = 0:N
    Ccurr = Ck{k+1};
    Dcurr = Dk{k+1};
    dlcurr = dlk{k+1};
    ducurr = duk{k+1};
    nC = size(Ccurr,1);
    
    if k == 0
        % At k = 0, v0 is known.
        Av = zeros(nC, N*m_u);
        offset = Ccurr * v0;
        Au = zeros(nC, N*m_u);
        Au(:, 1:m_u) = Dcurr;
    elseif k == N
        % Terminal state: v_N appears in the last block of V.
        rowV = (k-1)*n_v + 1 : k*n_v;
        Av = Ccurr * G(rowV, :);
        Au = zeros(nC, N*m_u);
        offset = Ccurr*(F(rowV,:)*v0 + w(rowV));
    else
        % For k = 1,...,N-1:
        rowV = (k-1)*n_v + 1 : k*n_v;
        Av = Ccurr * G(rowV, :);
        Au = zeros(nC, N*m_u);
        colU = k*m_u + 1 : (k+1)*m_u;
        Au(:, colU) = Dcurr;
        offset = Ccurr*(F(rowV,:)*v0 + w(rowV));
    end
    
    A_stage = Av + Au;
    lower = dlcurr - offset;
    upper = ducurr - offset;
    
    % Formulate the constraints:
    Aineq(rowStart:rowStart+nC-1, :) = A_stage;
    bineq(rowStart:rowStart+nC-1) = upper;
    Aineq(rowStart+nC:rowStart+2*nC-1, :) = -A_stage;
    bineq(rowStart+nC:rowStart+2*nC-1) = -lower;
    
    rowStart = rowStart + 2*nC;
end

%% Set lower and upper bounds on U (if applicable)
lb = [];
ub = [];

end  % end of mpc2qp

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Helper functions for time-varying systems (naive (no vectorization etc) implementations)
function Frow = buildFrowTimeVary(Acell, i, n_v)
% Returns the product A_{i} * ... * A_1 that multiplies v0.
Frow = eye(n_v);
for kk = 1:i
    Frow = Acell{kk} * Frow;
end
end

function offsetVal = buildOffsetTimeVary(Acell, Bcell, ccell, v0, i, timeV)
% Computes the accumulated offset due to ccell in a time-varying system.
n_v = size(v0,1);
offsetVal = zeros(n_v,1);
Aprod = eye(n_v);
for kk = 1:i
    offsetVal = offsetVal + Aprod * ccell{kk};
    Aprod = Acell{kk} * Aprod;
end
end

function Gblock = buildGblockTimeVary(Acell, Bcell, j, i)
% Computes the influence of u_{j} on v_{i} in a time-varying system.
n_v = size(Acell{1},1);
if j == i
    Gblock = Bcell{i};
else
    Gblock = eye(n_v);
    for kk = j+1:i
        Gblock = Acell{kk} * Gblock;
    end
    Gblock = Gblock * Bcell{j};
end
end
