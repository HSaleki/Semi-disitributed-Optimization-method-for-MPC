function [x, lambda, mu] = main(Q,A,b,C,c)

n_Q = length(Q);
dim_C = size(C{1},1);
dim_x = size(Q{1},1);
dim_A = size(A{1},1);

local = cell(n_Q,1);
Sum_CbarFmCbart = zeros(dim_C);
Sum_CbartFmB = zeros(dim_C,1);

% Construct local variables and assemble and calculate matrices
% Communicate required matrices
for i = 1:n_Q
    local{i} = local_var(Q{i},A{i},b{i},C{i});
    Sum_CbarFmCbart = Sum_CbarFmCbart + local{i}.CbarFmCbart;
    Sum_CbartFmB = Sum_CbartFmB + local{i}.CbartFmB;
end

% Solve for coupled constraint multiplier on central node
coupled_mult = global_var(Sum_CbarFmCbart,Sum_CbartFmB,c);

% Value to be communicated to local nodes 
mu = coupled_mult.mu;

% Calculate local variables
z = zeros(dim_x+dim_A,n_Q);
for i = 1:n_Q
    z(:,i) = local{i}.comp(mu);
end

x = z(1:dim_x,:);
lambda = z(dim_x+1:end,:);