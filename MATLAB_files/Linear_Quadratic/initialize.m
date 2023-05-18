%% Initialize problem parameter
dim_x = 6;
n_Q = 2;
n_A = 2;
dim_A = 2;
n_C = 2;
dim_C = 2;
c = randi([0,20],n_C,1);
Q = cell(n_Q,1);

for i = 1:n_Q
    temp = randi([0,20],dim_x);
    temp = temp*temp.';
    Q{i} = temp;
end

A = cell(n_A,1);
b = cell(n_A,1);

for i = 1:n_A
    A{i} = randi([0,20],dim_A,dim_x);
    b{i} = randi([0,20],dim_A,1);
end

C = cell(n_C,1);
for i = 1:n_C
    C{i} = randi([0,20],dim_C,dim_x);
end


%% Run main function solving the problem

[x, lambda,mu] = main(Q,A,b,C,c);

%% Quad Prog for reference

H = [Q{1}, zeros(dim_x); zeros(dim_x), Q{2}];
Aeq = [A{1}, zeros(dim_A,dim_x); zeros(dim_A,dim_x), A{2}; C{1}, C{2}];
beq = [b{1}; b{2}; c];
res = quadprog(H,[],[],[],Aeq,beq);
