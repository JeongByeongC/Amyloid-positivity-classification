clc; clear all; close all;
%% testing KNN classifier
train_files = csvread('./SUVR68_train.csv');
load('./meta_210210.mat');
train_id_from_csv = train_files(:, 1);
train_id = cell(length(train_id_from_csv), 1);

for i=1:length(train_id_from_csv)
    train_id{i} = sprintf('%08d', train_id_from_csv(i));
end
train_idx = find(ismember(meta.ID, train_id));
test_idx = setdiff(1:1:227, train_idx)';

%% louvein q=1.06 -> group 3, 0.9 -> group 2, GT is RCTU
GT_label = meta.BAPL;
GT_label(GT_label == 1) = 0; % amyloid negative
GT_label(GT_label == 2 | GT_label == 3) = 1; % amyloid positive

[y_new_group, group_prob] = Louvain_classfier(meta.suvr68, 0.9, 10000, false, true);
y_new_group(y_new_group == 2) = 0; % gender 2 = felmale

%% Find optimal K with hold out cross-validation method

intK = [1:2:51];
nK = numel(intK);

n = size(train_idx, 1);

train_data = [y_new_group(train_idx, :), meta.suvr68(train_idx, :)];
test_data = [y_new_group(test_idx, :), meta.suvr68(test_idx, :)];

[train, val] = KFold(train_data, 10);

Tm = zeros(1, nK);
Ts = zeros(1, nK);
Vm = zeros(1, nK);
Vs = zeros(1, nK);
train_pred = cell(10, nK);
train_pred_wrong = cell(10, nK);
val_pred = cell(10, nK);
val_pred_wrong = cell(10, nK);

for fold=1:10
    Xtr = train{fold}(:, 2:end);
    Ytr = train{fold}(:, 1);
    Xval = val{fold}(:, 2:end);
    Yval = val{fold}(:, 1);
    
    ik = 0;
    for k=intK
        ik = ik + 1;
        trYpred = KNNpredictor(Xtr, Ytr, Xtr, k);
        train_pred{fold, ik} = trYpred;
        wrongIdx = find(Ytr ~= trYpred);
        a = mean(Xtr(wrongIdx, :), 2);
        b = Ytr(wrongIdx);
        c = trYpred(wrongIdx);
        train_pred_wrong{fold, ik} = [wrongIdx, b, c, a];
        TrErr = immse(Ytr, trYpred);
        Tm(1, ik) = Tm(1, ik) + TrErr;
        Ts(1, ik) = Ts(1, ik) + TrErr^2;
        
        valYpred = KNNpredictor(Xtr, Ytr, Xval, k);
        val_pred{fold, ik} = valYpred;
        wrongIdx = find(Yval ~= valYpred);
        a = mean(Xval(wrongIdx, :), 2);
        b = Yval(wrongIdx);
        c = valYpred(wrongIdx);
        val_pred_wrong{fold, ik} = [wrongIdx, b, c, a];
        ValErr = immse(Yval, valYpred);
        Vm(1, ik) = Vm(1, ik) + ValErr;
        Vs(1, ik) = Vm(1, ik) + ValErr^2;
    end
end

Tm = Tm/10;
Ts = Vs/10 - Tm.^2;

Vm = Vm/10;
Vs = Vs/10 - Vm.^2;

optimal_K = min(intK(mean(Vm) == min(mean(Vm(:)))));

figure(3);
plot(intK, Tm, 'b-o');
hold on;
plot(intK, Vm, 'r-o');
axis tight;
line([optimal_K, optimal_K], [0 1], 'Color', 'k')
legend({'train', 'validation', 'Optimal K'});
hold off;
ylim([0, 0.05]);
xlim([0, 52]);
% close(findall(0, 'type', 'figure'));

%% Test data accuracy and MSE
Y_train = train_data(:, 1);
X_train = train_data(:, 2:end);
Y_test = test_data(:, 1);
X_test = test_data(:, 2:end);
Y_test(Y_test == 2) = 0;

test_pred = KNNpredictor(X_train, Y_train, X_test, 1);
test_pred(test_pred == 2) = 0;

acc = length(find(Y_test == test_pred))/length(Y_test) * 100; % wrong 1

right = X_test(Y_test == test_pred, :);
right_label = Y_test(Y_test == test_pred);
wrong = X_test(Y_test ~= test_pred, :);
wrong_label = Y_test(Y_test ~= test_pred);

test_true_amy_neg = mean(right(right_label == 0, :), 2);
TN = size(test_true_amy_neg, 1);
test_false_amy_neg = mean(wrong(wrong_label == 1, :), 2);
FP = size(test_false_amy_neg, 1);
test_true_amy_pos = mean(right(right_label == 1, :), 2);
TP = size(test_true_amy_pos, 1);
test_false_amy_pos = mean(wrong(wrong_label == 0, :), 2);
FN = size(test_false_amy_pos, 1);

figure(4);
boxplot(mean(X_test,2), Y_test, 'Colors', 'k', 'Labels', {'Amyloid -', 'Amyloid +'}, 'notch', 'off', 'symbol', '');
hold on;
scatter(ones(length(test_true_amy_neg), 1), test_true_amy_neg, 'filled', 'b', 'MarkerFaceAlpha', 0.5, 'jitter', 'on', 'jitterAmount', 0.1);
scatter(ones(length(test_false_amy_neg), 1), test_false_amy_neg, 'x', 'b', 'LineWidth', 2, 'MarkerFaceAlpha', 0.5, 'jitter', 'on', 'jitterAmount', 0.1);
scatter(2*ones(length(test_true_amy_pos), 1), test_true_amy_pos, 'filled', 'r', 'MarkerFaceAlpha', 0.5, 'jitter', 'on', 'jitterAmount', 0.1);
scatter(2*ones(length(test_false_amy_pos), 1), test_false_amy_pos, 'x', 'r', 'LineWidth', 2, 'MarkerFaceAlpha', 0.5, 'jitter', 'on', 'jitterAmount', 0.1);
legend({'True negative', 'False positive', 'True positive', 'False negative'}, 'Location', 'northwest');
hold off;


acc = (TN + TP) / (TN + TP + FN + FP);
% test_roc = roc_curve(test_amy_neg, test_amy_pos, 3, 1);
sen = TP /(TP + FN);
spe = TN / (TN + FP);
F1 = (2*TP) / (2*TP + FP + FN);


%% wrong idx in each folds
% train_wrong_idx = cell(10, 1);
% val_wrong_idx = cell(10, 1);
% 
% for i=1:size(val_pred_wrong, 1)
%     tmp1 = [];
%     tmp2 = [];
%     for j=1:size(val_pred_wrong, 2)
%         if any([train_pred_wrong{i, j}; val_pred_wrong{i, j}])
%             if ~isempty(train_pred_wrong{i, j})
%                 tmp1 = [tmp1;train_pred_wrong{i, j}(:, 1)];
%             end
%             if ~isempty(val_pred_wrong{i, j})
%                 tmp2 = [tmp2;val_pred_wrong{i, j}(:, 1)];
%             end
%         end
%     end
%     if ~isempty(tmp1)
%         train_wrong_idx{i, 1} = unique(tmp1);
%     end
%     if ~isempty(tmp2)
%         val_wrong_idx{i, 1} = unique(tmp2);
%     end
% end
% 
% tmp_suvr = [];
% for i=1:10
%     tmp_suvr = [tmp_suvr;mean(train{i}(train_wrong_idx{i}, 2:end), 2)];
% end
% 
% figure(2);
% hold on;
% histogram(unique(tmp_suvr));
% hold off;
% 
% tmp_suvr = [];
% for i=1:10
%     tmp_suvr = [tmp_suvr;mean(val{i}(val_wrong_idx{i}, 2:end), 2)];
% end
% 
% figure(2);
% hold on;
% histogram(unique(tmp_suvr));
% legend({'Positive', 'Negative', 'Train wrong', 'Val wrong'});
% hold off;
