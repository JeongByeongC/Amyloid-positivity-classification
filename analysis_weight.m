clc; clear all; close all;

%% load files
W = double(readNPY('./update_adj_1weight.npy'));
W_soft = double(readNPY('./update_adj_2weight.npy'));
% updated_G = double(readNPY('./new_G.npy'));
load('./meta_220618.mat');
load('../DK_atlas_name.mat');
X = meta.suvr_82roi;


%% line plot weight
update_W = double(readNPY('./update_adj_1weight.npy'));
update_W_soft = double(readNPY('./update_adj_2weight.npy'));

wo_W = double(readNPY('./no_update_XSUVR_1weight.npy'));
wo_W_soft = double(readNPY('./no_update_XSUVR_2weight.npy'));

cortex_update_w = update_W([1:34, 42:75], 1);
cortex_wo_w = wo_W([1:34, 42:75], 1);

max_val = max(max(update_W(:)), max(wo_W(:)));
min_val = min(min(update_W(:)), min(wo_W(:)));

c_max_val = max(max(cortex_update_w), max(cortex_wo_w));
c_min_val = min(min(cortex_update_w), min(cortex_wo_w));

norm_update_W = (update_W - min_val) ./ (max_val - min_val);
norm_wo_W = (wo_W - min_val) ./ (max_val - min_val);

norm_c_update_w = (cortex_update_w - c_min_val) ./ (c_max_val - c_min_val);
norm_c_wo_w = (cortex_wo_w - c_min_val) ./ (c_max_val - c_min_val);

x = 1:1:82;

plot(x, norm_update_W(:, 1), 'r');
hold on;
% plot(x, norm_update_W(:, 2), 'g');
plot(x, norm_wo_W, 'b');
legend({'proposed ch1', 'baseline1 ch1'}, 'Location', 'southwest');
ylim([0, 1]);
xlim([1, 82]);
xticks(x);
hold off;
set(gca, 'XTickLabel', DK_atlas_name);
xtickangle(45);

yyaxis left
plot(x, update_W(:, 1), '-or', 'MarkerFaceColor', 'r');
ylabel('Proposed CH 1');
ylim([min_val, max_val]);
yyaxis right
plot(x, wo_W, '-ob', 'MarkerFaceColor', 'b');
ylabel('Baseline 1 CH 1');
ylim([min_val, max_val]);
ax = gca;
ax.YAxis(1).Color = 'r';
ax.YAxis(2).Color = 'b';
xlim([1, 82]);
xticks(x);
xtickangle(50);
ax.XTickLabel = DK_atlas_name;
set(gca, 'FontSize', 12, 'FontWeight', 'bold');
box off;


%% brain overlay
c_norm_update_w = ceil((norm_c_update_w).*63)+1;
script_surface_coloring_bc('proposed_ch1', c_norm_update_w, './', 0);
close all;

c_norm_no_w = ceil((norm_c_wo_w).*63) + 1;
script_surface_coloring_bc('baseline_ch1', c_norm_no_w, './', 0);

%% softmax output weight
out_update_W = update_W * update_W_soft;
out_wo_W = wo_W * wo_W_soft;

max_val = max(max(out_update_W(:)), max(out_wo_W(:)));
min_val = min(min(out_update_W(:)), min(out_wo_W(:)));

x = 1:1:82;
yyaxis left
plot(x, out_update_W(:, 2), '-or', 'MarkerFaceColor', 'r');
ylabel('Proposed amyloid +');
ylim([min_val, max_val]);
yyaxis right
plot(x, out_wo_W(:, 2), '-ob', 'MarkerFaceColor', 'b');
ylabel('Baseline 1 amyloid +');
ylim([min_val, max_val]);
ax = gca;
ax.YAxis(1).Color = 'r';
ax.YAxis(2).Color = 'b';
xlim([1, 82]);
xticks(x);
xtickangle(50);
ax.XTickLabel = DK_atlas_name;
set(gca, 'FontSize', 12, 'FontWeight', 'bold');
box off;
%%
% new_X = X * W;
m = mean(W);
s = std(W);
cutoff = m + 1*s;
new_W = W;
new_W(new_W < cutoff) = 0;

for i=1:size(W, 2)
    tmp = new_W(:, i);
    roi_idx = find(tmp);
    node_idx = [1 (find(tmp)'+1)];
    whole_idx = 1:1:83;
    no_idx = setdiff(whole_idx, node_idx);


    ch1_A = zeros(83, 83);
    ch1_A(1, 2:end) = tmp;
    ch1_A(:, 1) = ch1_A(1, :);

    node_name = DK_atlas_name(roi_idx);
    hue = cell(1, length(node_name) + 1);
    hue{1, 1} = sprintf('ch%d', i);
    for j=2:length(node_name) + 1
        hue{1, j} = char(node_name{j-1});
    end
    figure(i);
    G = graph(ch1_A);
    G = rmnode(G, no_idx);
    h = plot(G, 'NodeLabel', hue, 'EdgeColor', [0 0 0], 'MarkerSize', 5);
    h.NodeFontSize = 12;
end

%% postivie = 2
W_output = W * W_soft;
m = mean(W_output);
s = std(W_output);
cutoff = m + 1*s;
W_output_thr = W_output;
W_output_thr(W_output_thr < cutoff) = 0;

for i=1:size(W_output_thr, 2)
    tmp = W_output_thr(:, i);
    roi_idx = find(tmp);
    node_idx = [1 (find(tmp)'+1)];
    whole_idx = 1:1:83;
    no_idx = setdiff(whole_idx, node_idx);


    ch1_A = zeros(83, 83);
    ch1_A(1, 2:end) = tmp;
    ch1_A(:, 1) = ch1_A(1, :);

    node_name = DK_atlas_name(roi_idx);
    hue = cell(1, length(node_name) + 1);
    if i == 1
        hue{1, 1} = 'amyloid -';
    else
        hue{1, 1} = 'amyloid +';
    end
    
    for j=2:length(node_name) + 1
        hue{1, j} = char(node_name{j-1});
    end
    figure(i*6);
    G = graph(ch1_A);
    G = rmnode(G, no_idx);
    h = plot(G, 'NodeLabel', hue, 'EdgeColor', [0 0 0], 'MarkerSize', 5);
    h.NodeFontSize = 12;
end