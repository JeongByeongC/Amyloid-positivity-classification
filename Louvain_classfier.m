function [ci, group_prob] =  Louvain_classfier(Xtr, gamma, iter, subtype, display)

ed_mat = zeros(size(Xtr, 1));

for i=1:length(ed_mat)
    for j=i:length(ed_mat)
        ed_mat(i, j) = sqrt(sum((Xtr(i, :) - Xtr(j, :)).^2));
        ed_mat(j, i) = ed_mat(i, j);
    end
end

norm_ed_mat = 1./(1.+ed_mat);

if subtype
    group_prob = zeros(size(Xtr, 1), 3);
    for i=1:iter
        [ci, q] = community_louvain(norm_ed_mat, gamma);
        g1_idx = find(ci == 1);
        g2_idx = find(ci == 2);
        g3_idx = find(ci == 3);
        group_prob(g1_idx, 1) = group_prob(g1_idx, 1) + 1;
        group_prob(g2_idx, 2) = group_prob(g2_idx, 2) + 1;
        group_prob(g3_idx, 3) = group_prob(g3_idx, 3) + 1;
    end
    
    group_prob = group_prob/iter;
    [~, ci] = max(group_prob, [], 2);
%     [sorted, idx] = sort(mean_suvr);
    
    [~, idx] = sort(ci);
    sorted_ed_mat = norm_ed_mat(idx, idx);
else
    group_prob = zeros(size(Xtr, 1), 2);
    for i=1:iter
        [ci, q] = community_louvain(norm_ed_mat, gamma);
        g1_idx = find(ci == 1);
        g2_idx = find(ci == 2);
        group_prob(g1_idx, 1) = group_prob(g1_idx, 1) + 1;
        group_prob(g2_idx, 2) = group_prob(g2_idx, 2) + 1;
    end
    
    group_prob = group_prob/iter;
    [~, ci] = max(group_prob, [], 2);
    [~, idx] = sort(ci);
    sorted_ed_mat = norm_ed_mat(idx, idx);
end
    

if display
    figure(1);
    imagesc(sorted_ed_mat, [0, 1]);
    colorbar;
    figure(3);
    imagesc(norm_ed_mat, [0, 1]);colorbar;
end
end