classdef local_var2
    % get functions f, g, matrix A, and vector b
    % Form Lagrangian and its gradient and Hessian
    % Form matrices required for KKT matrix, violation, stepsize selection
    properties (Access = private) 
        F
        A
        dim_A
        Dmr
        DmAbarT
    end
    properties
        Abar
        r
        delta_x_
        D
        nu_
        mu_
        gradF
        gradF_
        JacG
        JacG_
        JacH
        JacH_
        gradL_
        F_
        HesF
        HesF_
        HesG
        HesG_
        HesH
        HesH_
        G
        G_
        H
        H_
        Ax
        AbarDmr
        AbarDmAbarT
        dim_x
        x_
        lambda_
        s_
        gradFTdeltax_
        Falpha
        Galpha
        Halpha
        Aalpha
        delta_mu
        delta_nu
        dim_G
        dim_H
        delta_lambda_
        z_
        delta_z_
        KKT_vio_
    end
    methods
        function obj = local_var2(varargin)
            if nargin == 7
                if sum(cellfun(@isempty,varargin)) == 0
                    obj.F = varargin{1};
                    obj.G = varargin{2};
                    obj.H = varargin{3};
                    obj.A = varargin{4};
                    obj.nu_ = varargin{5};
                    obj.dim_A = size(obj.A,1);
                    obj.dim_x = varargin{6};
                    obj.x_ = varargin{7};
                    x = casadi.MX.sym('x',obj.dim_x,1);
                    obj.dim_G = length(obj.G(x));
                    obj.lambda_ = ones(obj.dim_G,1);
                    obj.dim_H = length(obj.H(x));
                    obj.mu_ = ones(obj.dim_H,1);
                    obj.s_ = ones(obj.dim_H,1);
                elseif not(isempty(varargin{2})) && isempty(varargin{4}) && isempty(varargin{5})
                    obj.F = varargin{1};
                    obj.G = varargin{2};
                    obj.H = varargin{3};
                    obj.A = [];
                    obj.nu_ = [];
                    obj.dim_x = varargin{6};
                    x = casadi.MX.sym('x',obj.dim_x,1);
                    obj.x_ = varargin{7};
                    obj.dim_G = length(obj.G(x));
                    obj.lambda_ = ones(obj.dim_G,1);
                    obj.dim_H = length(obj.H(x));
                    obj.mu_ = ones(obj.dim_H,1);
                    obj.s_ = ones(obj.dim_H,1);
                elseif isempty(varargin{2}) && isempty(varargin{3}) && isempty(varargin{4}) && isempty(varargin{5})
                    obj.F = varargin{1};
                    obj.G = [];
                    obj.H = [];
                    obj.lambda_ = [];
                    obj.mu_ = [];
                    obj.s_ = [];
                    obj.A = [];
                    obj.nu_ = [];
                    obj.dim_x = varargin{5};
                    obj.x_ = varargin{6};

                elseif not(isempty(varargin{2})) && isempty(varargin{3}) && isempty(varargin{4}) && isempty(varargin{5})
                    obj.F = varargin{1};
                    obj.G = varargin{2};
                    obj.H = [];
                    obj.A = [];
                    obj.nu_ = [];
                    obj.dim_x = varargin{6};
                    x = casadi.MX.sym('x',obj.dim_x,1);
                    obj.x_ = varargin{7};
                    obj.dim_G = length(obj.G(x));
                    obj.lambda_ = ones(obj.dim_G,1);
                    obj.mu_ = [];
                    obj.s_ = [];
                else
                    error('Wrong input')
                end
            else
                error('Wrong input')
            end
            %obj.x_ = zeros(dim_x,1);
            %obj.x_ = randi([1,5],obj.dim_x,1);
            obj.z_ = [obj.x_; obj.lambda_; obj.mu_; obj.s_];
            obj.gradF = obj.F.factory('gradF',{'x'},{'jac:y:x'});
            obj.JacG = obj.G.factory('JacG',{'x'},{'jac:y:x'});
            obj.HesF = obj.gradF.factory('HesF',{'x'},{'jac:jac_y_x:x'});
            obj.HesG = obj.JacG.factory('HesG',{'x'},{'jac:jac_y_x:x'});
            obj.JacH = obj.H.factory('JacH',{'x'},{'jac:y:x'});
            obj.HesH = obj.JacH.factory('HesH',{'x'},{'jac:jac_y_x:x'});
        end

        % function for evaluationg casadi fucntions and converting 
        % to matlab matrices for computation
        function obj = assemble(obj,tau_)
            gradF_ = obj.gradF(obj.x_);
            JacG_ = obj.JacG(obj.x_);
            HesF_ = obj.HesF(obj.x_);
            HesG_ = obj.HesG(obj.x_);
            G_ = obj.G(obj.x_);
            JacH_ = obj.JacH(obj.x_);
            HesH_ = obj.HesH(obj.x_);
            H_ = obj.H(obj.x_);
            obj.F_ = full(obj.F(obj.x_));
            obj.gradF_ = full(gradF_)';
            obj.JacG_ = full(JacG_);
            obj.HesF_ = full(HesF_);
            obj.HesG_ = full(HesG_);
            obj.G_ = full(G_);
            obj.JacH_ = full(JacH_);
            obj.HesH_ = full(HesH_);
            obj.H_ = full(H_);
            if isempty(obj.A)
                obj.gradL_ = obj.gradF_ + obj.JacG_.'*obj.lambda_ + obj.JacH_.'*obj.mu_;
            else
                obj.gradL_ = obj.gradF_ + obj.JacG_.'*obj.lambda_ + obj.JacH_.'*obj.mu_ + obj.A.'*obj.nu_;
            end
            temp = zeros(obj.dim_x);
            for i = 1:obj.dim_G
                temp = temp + obj.lambda_(i)*obj.HesG_((i-1)*obj.dim_x+1:i*obj.dim_x,:);
            end
            temp2 = zeros(obj.dim_x);
            for i = 1:obj.dim_H
                temp2 = temp2 + obj.mu_(i)*obj.HesH_((i-1)*obj.dim_x+1:i*obj.dim_x,:);
            end
            HesL_ = obj.HesF_ + temp + temp2;
            HesL_reg = hessian_regularization(HesL_);
            obj.D = [HesL_reg, obj.JacG_.', obj.JacH_.', zeros(obj.dim_x,obj.dim_H); obj.JacG_, zeros(obj.dim_G,obj.dim_G+2*obj.dim_H);
                obj.JacH_, zeros(obj.dim_H,obj.dim_G+obj.dim_H), eye(obj.dim_H); zeros(obj.dim_H,obj.dim_x+obj.dim_G),eye(obj.dim_H), diag(obj.s_)\diag(obj.mu_)];
%             if rank(obj.D) < length(obj.D)
%                 obj.D = obj.D + 1e-4*eye(length(obj.D));
%             end
%             obj.D = Hessian_reg();
            dD =  decomposition(obj.D,'lu');
            obj.r = [obj.gradL_; obj.G_; obj.H_ + obj.s_; diag(obj.s_)*obj.mu_ - ones(1,obj.dim_H)*tau_];
            obj.KKT_vio_ = max(norm(obj.r,inf) >= 1e-8);
            obj.Dmr = dD\obj.r;
            if not(isempty(obj.A))
                obj.Abar = [obj.A, zeros(obj.dim_A, obj.dim_G + 2*(obj.dim_H))];
                obj.AbarDmAbarT = obj.Abar*(dD\obj.Abar.');
                obj.AbarDmr = obj.Abar*(dD\obj.r);
                obj.DmAbarT = dD\(obj.Abar');
                obj.Ax = obj.A*obj.x_;
            end

           
        end
        function obj = solve(obj,delta_nu)
            %obj.delta_mu = delta_mu;
            if isempty(delta_nu)
                obj.delta_z_ = - obj.Dmr;
            else 
%             if not(isempty(obj.A))
                obj.delta_nu = delta_nu;
                obj.delta_z_ = - obj.Dmr - obj.DmAbarT*delta_nu;
%             else 
%                 obj.delta_z_ = - obj.Dmr;
            end
            obj.delta_x_ = obj.delta_z_(1:obj.dim_x);
            obj.delta_lambda_ = obj.delta_z_(obj.dim_x+1:obj.dim_x+obj.dim_G);
            obj.delta_mu_ = obj.delta_z_(obj.dim_x+obj.dim_G+1:obj.dim_x+obj.dim_G+obj.dim_H);
            obj.delta_s_ = obj.delta_z_(obj.dim_x+obj.dim_G+obj.dim_H+1:end);
            obj.gradFTdeltax_ = obj.gradF_.'*obj.delta_x_;

        end
        function output = merit(obj,alpha)
            Falpha = obj.F(obj.x_ + alpha*obj.delta_x_);
            Galpha = obj.G(obj.x_ + alpha*obj.delta_x_);
            Halpha = obj.H(obj.x_ + alpha*obj.delta_x_);
            Falpha_ = full(Falpha);
            Galpha_ = full(Galpha);
            Halpha_ = full(Halpha)+ obj.s_+alpha*obj.delta_s_;
            if not(isempty(obj.A))
                Aalpha = obj.A*(obj.x_ + alpha*obj.delta_x_);
                Aalpha_ = full(Aalpha);
                output = {Falpha_, Galpha_, Halpha_, Aalpha_};
            else 
                output = {Falpha_, Galpha_, Halpha_};
            end
            
        end
        function obj = update(obj,alpha)
            obj.z_ = obj.z_ + alpha* obj.delta_z_;
            obj.x_ = obj.z_(1:obj.dim_x);
            %obj.lambda_ = obj.z_(obj.dim_x+1:end);
            obj.delta_lambda_ = obj.delta_z_(obj.dim_x+1:obj.dim_x+obj.dim_G);
            obj.delta_mu_ = obj.delta_z_(obj.dim_x+obj.dim_G+1:obj.dim_x+obj.dim_G+obj.dim_H);
            obj.delta_s_ = obj.delta_z_(obj.dim_x+obj.dim_G+obj.dim_H+1:end);

            if not(isempty(obj.A))
                obj.nu_ = obj.nu_ + alpha*obj.delta_nu;
            end
        end
    end
end
