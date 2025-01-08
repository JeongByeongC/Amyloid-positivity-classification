clc; clear all; close all;
%%
train_files = csvread('../AMY_pos_neg_classification/SUVR68_train.csv');
% load('../AMY_pos_neg_classification/meta_210210.mat');
load('./meta_220618.mat');
train_id_from_csv = train_files(:, 1);
train_id = cell(length(train_id_from_csv), 1);

for i=1:length(train_id_from_csv)
    train_id{i} = sprintf('%08d', train_id_from_csv(i));
end
train_idx = find(ismember(meta.ID, train_id));
test_idx = setdiff(1:1:227, train_idx)';

train_suvr = meta.suvr_82roi(train_idx, :);
test_suvr = meta.suvr_82roi(test_idx, :);
%%
test_GT = readNPY('./test_GT.npy');
test_pred = readNPY('./no_update_XSUVR_test.npy');
train_GT = readNPY('./train_GT.npy');
train_pred = readNPY('./no_update_XSUVR_train.npy');

test_GT = test_GT + 1;
test_pred = test_pred + 1;
train_GT = train_GT + 1;
train_pred = train_pred + 1;

right = train_suvr(train_GT == train_pred, :);
right_label = train_GT(train_GT == train_pred);
wrong = train_suvr(train_GT ~= train_pred, :);
wrong_label = train_GT(train_GT ~= train_pred);

lou_cm_train = confusionmat(train_GT, train_pred);

lou_train_TN = mean(right(right_label==1, :), 2);
lou_train_TP = mean(right(right_label==2, :), 2);
lou_train_FN = mean(wrong(wrong_label == 2, :), 2);
lou_train_FP = mean(wrong(wrong_label == 1, :), 2);

TN = length(lou_train_TN);
TP = length(lou_train_TP);
FN = length(lou_train_FN);
FP = length(lou_train_FP);

acc = (TP + TN) / (TP + TN + FP + FN);
precision = TP / (TP+FP);
recall = TP / (TP + FN);

fprintf('Train Accuracy = %.4f, precision = %.4f, recall = %.4f \n', acc, precision, recall);

figure(119);
boxplot(mean(train_suvr, 2), train_GT, 'Colors', 'k', 'Labels', {'Amyloid -', 'Amyloid +'}, 'notch', 'off', 'symbol', '');
hold on;
scatter(ones(length(lou_train_TN), 1), lou_train_TN, 'filled', 'b', 'MarkerFaceAlpha', 0.5, 'jitter', 'on', 'jitterAmount', 0.1);
scatter(2*ones(length(lou_train_TP), 1), lou_train_TP, 'filled', 'r', 'MarkerFaceAlpha', 0.5, 'jitter', 'on', 'jitterAmount', 0.1);
scatter(ones(length(lou_train_FP), 1), lou_train_FP, 'x', 'b', 'LineWidth', 2, 'MarkerFaceAlpha', 0.5, 'jitter', 'on', 'jitterAmount', 0.1);
scatter(2*ones(length(lou_train_FN), 1), lou_train_FN, 'x', 'r', 'LineWidth', 2, 'MarkerFaceAlpha', 0.5, 'jitter', 'on', 'jitterAmount', 0.1);
legend({sprintf('True negative %d', length(lou_train_TN)), sprintf('True positive %d', length(lou_train_TP)), ...
    sprintf('False positive %d', length(lou_train_FP)), sprintf('False negative %d', length(lou_train_FN))}, 'Location', 'northwest');
hold off;
figure(120);
cm = confusionchart(lou_cm_train, categorical({'Amyloid -', 'Amyloid +'}));

test_right = test_suvr(test_GT == test_pred, :);
test_right_label = test_GT(test_GT == test_pred);
test_wrong = test_suvr(test_GT ~= test_pred, :);
test_wrong_label = test_GT(test_GT ~= test_pred);

lou_cm_test = confusionmat(test_GT, test_pred);

test_TN = mean(test_right(test_right_label==1, :), 2);
test_TP = mean(test_right(test_right_label==2, :), 2);
test_FN = mean(test_wrong(test_wrong_label == 2, :), 2);
test_FP = mean(test_wrong(test_wrong_label == 1, :), 2);

TN = length(test_TN);
TP = length(test_TP);
FN = length(test_FN);
FP = length(test_FP);

acc = (TP + TN) / (TP + TN + FP + FN);
precision = TP / (TP+FP);
recall = TP / (TP + FN);

fprintf('Test Accuracy = %.4f, precision = %.4f, recall = %.4f \n', acc, precision, recall);

figure(176);
boxplot(mean(test_suvr, 2), test_GT, 'Colors', 'k', 'Labels', {'Amyloid -', 'Amyloid +'}, 'notch', 'off', 'symbol', '');
hold on;
scatter(ones(length(test_TN), 1), test_TN, 'filled', 'b', 'MarkerFaceAlpha', 0.5, 'jitter', 'on', 'jitterAmount', 0.1);
scatter(2*ones(length(test_TP), 1), test_TP, 'filled', 'r', 'MarkerFaceAlpha', 0.5, 'jitter', 'on', 'jitterAmount', 0.1);
scatter(ones(length(test_FP), 1), test_FP, 'x', 'b', 'LineWidth', 2, 'MarkerFaceAlpha', 0.5, 'jitter', 'on', 'jitterAmount', 0.1);
scatter(2*ones(length(test_FN), 1), test_FN, 'x', 'r', 'LineWidth', 2, 'MarkerFaceAlpha', 0.5, 'jitter', 'on', 'jitterAmount', 0.1);
legend({sprintf('True negative %d', length(test_TN)), sprintf('True positive %d', length(test_TP)), ...
    sprintf('False positive %d', length(test_FP)), sprintf('False negative %d', length(test_FN))}, 'Location', 'northwest');
hold off;

figure(177);
cm = confusionchart(lou_cm_test, categorical({'Amyloid -', 'Amyloid +'}));