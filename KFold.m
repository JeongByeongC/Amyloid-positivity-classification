function [train_data, val_data] = KFold(data, n_split)
no_of_rows = size(data, 1);
train_data{n_split, 1} = [];
val_data{n_split, 1} = [];

fold = ceil(no_of_rows / n_split);
%i==7 >> fold = 22

train_data{1} = data(fold +1 : end, :);
val_data{1} = data(1:fold, :);

for i = 2:n_split
    fold = 18;
    train_data{i} = [data(1:(i-1)*fold, :); data(i*fold+1:end, :)];
    val_data{i} = data((i-1)*fold +1:i*fold, :);
end
end