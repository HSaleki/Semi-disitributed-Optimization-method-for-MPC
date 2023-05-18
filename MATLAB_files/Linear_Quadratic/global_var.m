classdef global_var
    properties 
        mu
        c
        Sum_CbarFmCbart
        Sum_CbartFmB
    end
    methods
        function obj = global_var(Sum_CbarFmCbart,Sum_CbartFmB,c)
            obj.Sum_CbarFmCbart = Sum_CbarFmCbart;
            obj.Sum_CbartFmB = Sum_CbartFmB;
            obj.c = c;
            obj.mu = obj.Sum_CbarFmCbart\(obj.Sum_CbartFmB - obj.c);
        end
    end
end


