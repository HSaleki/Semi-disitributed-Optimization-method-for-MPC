classdef local_var
    properties (Access = private)
        FmCbart
        FmB
    end
    properties
        CbarFmCbart
        CbartFmB
        mu
    end
    methods
        function obj = local_var(Q,A,b,C)
            F = [Q, A'; A, zeros(size(A,1))];
            dF = decomposition(F,'lu');
            Cbar = [C, zeros(size(C,1),size(A,1))];
            B = [zeros(size(C,2),1); b];
            obj.FmCbart = dF\Cbar.';
            obj.CbarFmCbart = Cbar*obj.FmCbart;
            obj.FmB = dF\B;
            obj.CbartFmB = Cbar*obj.FmB;
        end
        function z = comp(obj,mu)
            obj.mu = mu;
            z = obj.FmB - obj.FmCbart*obj.mu;
        end
    end
end


