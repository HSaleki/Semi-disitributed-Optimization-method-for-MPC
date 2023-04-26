clear
close all

%% Set the dimensions, initial values and matrices

tol = 1e-4;
n_x = 3;
x_1_init = randi([0,2],n_x,1);
x_2_init = randi([0,2],n_x,1);

Q1 = rand(n_x);
Q1 = Q1*Q1.';

Q2 = rand(n_x);
Q2 = Q2*Q2.';

n_A1 = 1;
n_A2 = 1;
n_C = 2;

A1 = randi([0,20],n_A1,n_x);
A2 = randi([0,20],n_A2,n_x);
b1 = randi([0,20],n_A1,1);
b2 = randi([0,20],n_A2,1);
C1 = randi([0,20],n_C,n_x);
C2 = randi([0,20],n_C,n_x);


%% Solution

% residuals
r1 = zeros(2*n_x,1);
r2 = [b1; b2];
r3 = zeros(n_C,1);
% modified residuals
r2_bar = r2;
r3_bar = r3;

% modified matrices to account for elimination of x
% [M1, M2; M3, M4][lambda; mu] = [r2_bar; r3_bar]
M1 = [-A1*(Q1\A1.'), zeros(n_A1,n_A2); zeros(n_A2,n_A1), -A2*(Q2\A2.')];
M2 = [-A1*(Q1\C1.'); + A2*(Q2\C2.')];
M3 = [-C1*(Q1\A1.'), C2*(Q2\A2.')];
M4 = -(C1*(Q1\C1.') + C2*(Q2\C2.'));

% decision variable and Lagrange multipliers
mu = - (M4 - M3*(M1\M2))\(M3*(M1\r2_bar) - r3_bar);
lambda1 = (M1(1:n_A1,1:n_A1))\(r2_bar(1:n_A1) - M2(1:n_A1,:)*mu);
lambda2 = (M1(n_A1+1:end,n_A1+1:end))\(r2_bar(n_A1+1:end) - M2(n_A1+1:end,:)*mu);
X1 = - Q1\(A1.'*lambda1 + C1.'*mu);
X2 = - Q2\(A2.'*lambda2 - C2.'*mu);



%% Optimization toolbox to check

x1 = optimvar('x1', n_x);
x2 = optimvar('x2', n_x);

cost = 0.5*(x1.'*Q1*x1 + x2.'*Q2*x2);
c1 = A1*x1 == b1;
c2 = A2*x2 == b2;
c3 = C1*x1 == C2*x2;

prob=optimproblem('Objective',cost,'ObjectiveSense','min');
prob.Constraints.c1=c1;
prob.Constraints.c2=c2;
prob.Constraints.c3=c3;

[sol,cost] = solve(prob);
