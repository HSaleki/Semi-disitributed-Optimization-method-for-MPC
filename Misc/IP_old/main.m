function [X, lambda, mu, nu] = main(varargin)

%varargout = main(varargin)
%[X, lambda, mu] = main(varargin)
%main(F,G,A,b,n_x,dim_x,varargin)

% indexing is done by number of x, if there isnt a corresponding f_i or g_i or h_i they should 
% be put as zero input

% CORRECT USAGES
% main(F,G,H,A,B,n_x,dim_x)
% main(F,G,H,[],[],n_x,dim_x)
% main(F,G,[],[],[],n_x,dim_x)
% main(F,[],[],[],[],n_x,dim_x)
% lambda corresponds to G, mu corresponds to H, local
% nu corresponds to coupled linear constraint

if nargin == 8
    if sum(cellfun(@isempty,varargin)) == 0
        F=varargin{1};
        G=varargin{2};
        H=varargin{3};
        A=varargin{4};
        b=varargin{5};
        n_x=varargin{6};
        dim_x=varargin{7};
        x_in = varargin{8};
    elseif not(isempty(varargin{2})) && isempty(varargin{3}) && isempty(varargin{4}) && isempty(varargin{5})
        F=varargin{1};
        G=varargin{2};
        H={};
        A={};
        b=[];
        mu =[];
        n_x=varargin{6};
        dim_x=varargin{7};
        x_in = varargin{8};

      elseif not(isempty(varargin{2})) && not(isempty(varargin{3})) && isempty(varargin{4}) && isempty(varargin{5})
        F=varargin{1};
        G=varargin{2};
        H=varargin{3};
        A={};
        b=[];
        mu = [];
        n_x=varargin{5};
        dim_x=varargin{6};
        x_in = varargin{7};

    elseif isempty(varargin{2}) && isempty(varargin{3}) && isempty(varargin{4}) && isempty(varargin{5})
        F=varargin{1};
        G={};
        H={};
        A={};
        b=[];
        mu = [];
        n_x=varargin{5};
        dim_x=varargin{6};
        x_in = varargin{7};
    else
        error('Wrong input')
    end
else
    error('Wrong input')
end

tol = 1e-8;
tau = 1;

% dim_x = size(A{1},2);
if not(isempty(A))
    dim_A = size(A{1},1);
    nu = ones(dim_A,1);
    Sum_AbarDmAbarT = zeros(dim_A);
    Sum_AbarDmr = zeros(dim_A,1);
    Sum_Ax = zeros(dim_A,1);
end

local = cell(n_x,1);
sigma = 2;
dim_G = zeros(n_x,1);
dim_H = dim_G;

gradL_ = zeros(n_x*dim_x,1);
F_ = 0;

% Construct local variables, calculate, and assemble matrices
% Communicate required matrices
for i = 1:n_x
    if not(isempty(A))
        local{i} = local_var2(F{i},G{i},H{i},A{i},nu,dim_x,x_in{i});
    else 
        local{i} = local_var2(F{i},G{i},H{i},[],[],dim_x,x_in{i});
    end
    dim_G(i) = local{i}.dim_G;
    dim_H(i) = local{i}.dim_H;
    local{i} = assemble(local{i},tau);
    if not(isempty(A))
        Sum_AbarDmAbarT = Sum_AbarDmAbarT + local{i}.AbarDmAbarT;
        Sum_AbarDmr = Sum_AbarDmr + local{i}.AbarDmr;
        Sum_Ax = Sum_Ax + local{i}.Ax;
    end
    gradL_((i-1)*dim_x+1:i*dim_x,1) = local{i}.gradL_;
    F_ = F_ + local{i}.F_;
end
G_ = zeros(sum(dim_G),1);
dim_G_ = [0; dim_G];
H_ = zeros(sum(dim_H),1);
dim_H_ = [0; dim_H];
S_ = zeros(sum(dim_H),1);
dim_S_ = [0; dim_S];
for i = 1:n_x
      G_(sum(dim_G_(1:i))+1:sum(dim_G_(1:i+1))) = local{i}.G_;
      H_(sum(dim_H_(1:i))+1:sum(dim_H_(1:i+1))) = local{i}.H_;
      S_(sum(dim_S_(1:i))+1:sum(dim_S_(1:i+1))) = local{i}.s_;
end

if not(isempty(A))
    KKT_violation = [gradL_; G_; H_+S_; (Sum_Ax -b)];
else 
    KKT_violation = [gradL_; G_; H_+S_];
end

KKT_vio = norm(KKT_violation,inf);
iter = 0;


%% Newton Step

while KKT_vio > tol && tau > tol
    % Solve for coupled constraint multiplier on central node
%     if rank(Sum_AbarDmAbarT) < length(Sum_AbarDmAbarT)
%         Sum_AbarDmAbarT = Sum_AbarDmAbarT + tol*eye(length(Sum_AbarDmAbarT));
%     end
    if not(isempty(A))
        delta_nu = Sum_AbarDmAbarT\(Sum_Ax - b - Sum_AbarDmr);
        for i = 1:n_x
            local{i} = local{i}.solve(delta_nu);
        end
    else 
        for i = 1:n_x
            local{i} = local{i}.solve([]);
        end
    end
    gradFTdeltax_ = 0;
    for i = 1:n_x
        gradFTdeltax_ = gradFTdeltax_ + local{i}.gradFTdeltax_;
    end

    if not(isempty(A))
        Phi_0 = F_ + sigma*(norm(G_,1)+norm(H_+S_,1)+norm((Sum_Ax -b),1));
        DPhi_0 = gradFTdeltax_ - sigma*(norm(G_,1)+norm(H_+S_,1)+norm((Sum_Ax -b),1));
    else 
        Phi_0 = F_ + sigma*(norm(G_,1)+norm(H_+S_,1));
        DPhi_0 = gradFTdeltax_ - sigma*(norm(G_,1)+norm(H_+S_,1));        
    end
    beta = 0.5;
    gamma = 1e-4;
    alpha_ = 1;
    f = 0;
    g = [];
    h = [];
    if not(isempty(A))
        a = zeros(dim_A,1);
    end
    for i = 1:n_x
        temp = local{i}.merit(alpha_);
        f = f + temp{1};
        g = [g;temp{2}];
        h = [h;temp{3}];
        if not(isempty(A))
            a = a + temp{4};
        end
    end

    if not(isempty(A))
        Phi_alpha = f + sigma*(norm(g,1)+norm(h,1)+norm((a -b),1));
    else
        Phi_alpha = f + sigma*(norm(g,1)+norm(h,1));
    end

    
    while Phi_alpha >= Phi_0 + gamma*alpha_*DPhi_0 && alpha_ > 1e-4
        alpha_ = beta*alpha_;
        f = 0;
        g = [];
        h = [];
        if not(isempty(A))
            a = zeros(dim_A,1);
        end
        
        for i = 1:n_x
            temp = local{i}.merit(alpha_);
            f = f + temp{1};
            g = [g;temp{2}];
            h = [h;temp{3}];
            if not(isempty(A))
                a = a + temp{4};
            end
        end
        if not(isempty(A))
            Phi_alpha = f + sigma*(norm(g,1)+norm(h,1)+norm((a -b),1));
        else
            Phi_alpha = f + sigma*(norm(g,1)+norm(h,1));
        end
    end
    if not(isempty(A))
        nu = nu + alpha_*delta_nu;
    end
    % update local and global variables using alpha
    for i = 1:n_x
        local{i} = local{i}.update(alpha_);
    end

    % decrease the barrier parameter
    if (KKT_vio <= tau) 
        tau = max(0.1*tau, tol);
    end

    if not(isempty(A))
        Sum_AbarDmAbarT = zeros(dim_A);
        Sum_AbarDmr = zeros(dim_A,1);
        Sum_Ax = zeros(dim_A,1);
    end
    gradL_ = zeros(n_x*dim_x,1);
    G_ = zeros(sum(dim_G),1);
    H_ = zeros(sum(dim_H),1);
    F_ = 0;
    for i = 1:n_x
        local{i} = assemble(local{i},tau);
        if not(isempty(A))
            Sum_AbarDmAbarT = Sum_AbarDmAbarT + local{i}.AbarDmAbarT;
            Sum_AbarDmr = Sum_AbarDmr + local{i}.AbarDmr;
            Sum_Ax = Sum_Ax + local{i}.Ax;
        end
        gradL_((i-1)*dim_x+1:i*dim_x,1) = local{i}.gradL_;
        F_ = F_ + local{i}.F_;
    end
    
    for i = 1:n_x
        G_(sum(dim_G_(1:i))+1:sum(dim_G_(1:i+1))) = local{i}.G_;
        H_(sum(dim_H_(1:i))+1:sum(dim_H_(1:i+1))) = local{i}.H_;
        S_(sum(dim_S_(1:i))+1:sum(dim_S_(1:i+1))) = local{i}.s_;
    end
    
    
    if not(isempty(A))
        KKT_violation = [gradL_; G_; H_+S_; (Sum_Ax -b)];
    else 
        KKT_violation = [gradL_; G_ H_+S_];
    end
    KKT_vio = norm(KKT_violation,inf);
    X = zeros(n_x*dim_x,1);
    lambda = [];
    mu = [];
    for i = 1:n_x
        X((i-1)*dim_x+1:i*dim_x,1) = local{i}.x_;
        lambda = [lambda; local{i}.lambda_];
        mu = [mu; local{i}.mu_];
    end

    
    % change value of sigma
    if not(isempty(A))
        if (2*norm([lambda;mu;nu],inf)> sigma) 
            sigma = 2*norm([lambda;mu;nu],inf);
        end
    else
        if (2*norm([lambda;mu],inf)> sigma) 
            sigma = 2*norm([lambda;mu],inf);
        end
    end
    iter = iter + 1;
%     if ~isempty(lambda) && ~isempty(mu)
%         varargout = {X,lambda,mu};
%     elseif isempty(lambda) && ~isempty(mu)
%         varargout = {X,mu};
%     elseif not(isempty(lambda)) && isempty(mu)
%         varargout = {X,lambda};
%     elseif isempty(lambda) && isempty(mu)
%         varargout = {X};
%     end
end

