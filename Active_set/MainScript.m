% MainScript.m
clear;
clc;
% Main script for loading the data and initiating the algorithm
% The problem to solve is the following linea problem
% Using active set method
%       min_x sum_i (1/2 x_i' Q_i x_i + p_i' x_i)
% subject to: G_i x_i = g_i
%             h_i^{l} <= H_i x_i <= h_i^{u}
%             sum_i lb<= A_i x_i <= ub

% Step 1: Load Data from MAT-File
data = load('data2.mat'); % Load the .mat file containing the cell arrays
% Determine the number of local nodes (assumes H is always present)
if isfield(data, 'Q') && ~isempty(data.Q)
    N = numel(data.Q);
else
    error('Missing or empty H matrix in data.mat.');
end
dec_var = cell(1, N);
% Step 2: Ensure All Variables Exist and Are Non-Empty
fields_to_check = {'p', 'G', 'g', 'H', 'hl', 'hu', 'A', 'lb', 'ub'};
default_values = {cell(1, N), cell(1, N), cell(1, N), cell(1, N), cell(1, N), cell(1, N), cell(1, N), {}, {}};

for k = 1:length(fields_to_check)
    field = fields_to_check{k};
    if ~isfield(data, field) || isempty(data.(field))
%         fprintf('Warning: Missing or empty %s in data.mat. Using default values.\n', field);
        data.(field) = default_values{k}; % Assign default value
    end
end

% Extract variables from the loaded data
Q = data.Q;
p = data.p;
G = data.G;
g = data.g;
H = data.H;
hl = data.hl;
hu = data.hu;
A = data.A;
lb = data.lb;
ub = data.ub;

% Step 3: Initialize Local Nodes
nodes = cell(1, N); % Preallocate cell array for LocalNode objects
for i = 1:N
    nodes{i} = LocalNode(i, Q{i}, p{i}, G{i}, g{i}, H{i}, hl{i}, hu{i}, A{i});
end

% Step 4: Initialize Central Node
central = CentralNode(nodes, cell2mat(lb), cell2mat(ub));

% Step 5: Optimization Loop
max_iters = 100;
tolerance = 1e-6;
for iter = 1:max_iters

    % Receive matrices and vectors form local nodes and compute delta_mu
    delta_mu = central.aggregateAndSolveDeltaMu();
    % Broadcast delta_mu to local nodes
    central.broadcastDeltaMu();
    % Receive Ax and Adeltax and delta tau from local nodes to determine
    % the active set
    central.aggregateAndSolvestep();

    exit_flag = central.homotopystep();
    if exit_flag == 1
        % Retrieve decision variables
        for i = 1:N
            dec_var{i} = nodes{i}.getdecisionvariable;
        end
        break;
    end

end
