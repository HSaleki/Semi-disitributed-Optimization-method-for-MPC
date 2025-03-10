function HesL_reg = hessian_regularization(HesL_)

epsilon = 1e-6;
[E, L] = eig(HesL_);
L_ = L;
for i = 1:length(L_)
    if L_(i,i) < epsilon
        L_(i,i) = epsilon;
    end
end
HesL_reg = HesL_ + E*(L_-L)*E.';
end