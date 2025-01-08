function Ypred = KNNpredictor(Xtr, Yfrom_sim, Xte, k)
%
% Input parameters
%   Xtr = training input data
%   Ytr = training label data from Louvain method
%   Xte = test input data
%   k = number of neighbours from CV method odd is prefered
%
% Output parameters 
%   Ypred = prediction of test data

%%
n = size(Xtr, 1);
if k > n
    k=n;
end

test_sim_mat = zeros(size(Xte, 1), size(Xtr, 1));

for i=1:size(test_sim_mat, 1)
    for j=1:size(test_sim_mat, 2)
        test_sim_mat(i, j) = sqrt(sum((Xte(i, :) - Xtr(j, :)).^2));
    end
end

norm_test_sim_mat = 1./(1+test_sim_mat);
[a, idx] = sort(norm_test_sim_mat, 2, 'descend');
K_idx = idx(:, 1:k);

Ypred = mode(Yfrom_sim(K_idx), 2);
% Ypred = zeros(size(Xte, 1), 2);
% for i=1:size(K_idx, 1)
%     tmp = Yfrom_sim(K_idx(1, :));
%     g1_prob = length(find(tmp == 1))/k;
%     g2_prob = length(find(tmp == 2))/k;
%     Ypred(i, :) = [g1_prob, g2_prob];
% end
end
