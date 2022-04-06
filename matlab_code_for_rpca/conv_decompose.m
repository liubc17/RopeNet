function [rank] = conv_decompose(weight, lambda, savename, bias)
%CONV_DECOMPOSE 

[m,n,p,q] = size(weight);
weight = permute(weight,[1,4,3,2]);
weight = reshape(weight,[m, n * p * q]);
[rank, U, S, V, E] = svd_rpca(weight, lambda);
US = U * S;
[E_5, E_6, E_7, E_8, E_9, E_95, E_98, E_99] = deal(E);
sort_E = sort(abs(E(:)),1);
breakpoint5 = sort_E(round(length(sort_E) * 0.5));
breakpoint6 = sort_E(round(length(sort_E) * 0.6));
breakpoint7 = sort_E(round(length(sort_E) * 0.7));
breakpoint8 = sort_E(round(length(sort_E) * 0.8));
breakpoint9 = sort_E(round(length(sort_E) * 0.9));
breakpoint95 = sort_E(round(length(sort_E) * 0.95));
breakpoint98 = sort_E(round(length(sort_E) * 0.98));
breakpoint99 = sort_E(round(length(sort_E) * 0.99));
E_5(abs(E_5)<breakpoint5)=0;
E_6(abs(E_6)<breakpoint6)=0;
E_7(abs(E_7)<breakpoint7)=0;
E_8(abs(E_8)<breakpoint8)=0;
E_9(abs(E_9)<breakpoint9)=0;
E_95(abs(E_95)<breakpoint95)=0;
E_98(abs(E_98)<breakpoint98)=0;
E_99(abs(E_99)<breakpoint99)=0;
if nargin < 4
    save([savename,'.mat'],'US','V','E','E_5','E_6','E_7','E_8','E_9','E_95','E_98','E_99','lambda');
else
    save([savename,'.mat'],'US','V','E','E_5','E_6','E_7','E_8','E_9','E_95','E_98','E_99','lambda','bias');
end
end