function [val, index] = RT(u, v, eps_den, eps_num, w)

        % A function to perform ratio test for finding step lenth
        % Inputs:
        %   u - vector of numerators
        %   v - vector of denominators
        %   w - vector of inactive inequalities
        % Output:
        % [val, index] = RT(u,v) = min_i{u_i/v_i | 1 <= i <= m, v_i > 0}

        u_cut = max(u, 0);

        Mask = (v >= eps_den) & (u_cut >= eps_num);

        if ~any(Mask)
            % No valid constraints
            val    = Inf;
            index = NaN;
            return;
        end
        
        if isempty(w)
            vec = (1:length(u))';
            validIndices = vec(Mask); 
            validRatios = u_cut(Mask) ./ v(Mask);
            [val, ind] = min(validRatios);
            index = validIndices(ind);
        else
            validIndices = w(Mask);
            validRatios = u_cut(Mask) ./ v(Mask);
            [val, ind] = min(validRatios);
            index = validIndices(ind);
        end
end