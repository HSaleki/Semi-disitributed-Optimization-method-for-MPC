function val = assignIfEmpty(input, n, flag)
    if ~isempty(input)
        val = input;
    else
        val = flag*ones(n,1)*1e8;
    end
end