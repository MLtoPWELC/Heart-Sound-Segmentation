clear all;close all;

temp_path = [path{path_num},set{train_or_test},'.npy'];%path of TCN predict results
b = readNPY(temp_path);
a = a(index_use);
maxlen_0 = 1826;%1826 1772
num = length(index_use);
result = {};
LEN_sound = [];
right_num = [];
preds = {};

for i = 1:num
    temp = reshape(b(i,:,:),maxlen_0, 4);
    flag = length(a{i});
    LEN_sound = [LEN_sound flag];
    temp = temp(end-flag+1:end,:);
    [~,Ind] = max(temp');
    right_num = [right_num sum(Ind==a{i})];
    result{i} = temp;
    preds{i} = Ind;
end

acc_all = sum(right_num)/sum(LEN_sound);

preds_news_test = {};
signal = signal(index_use);
acc_viterbitest_all = [];
right_viterbi_num = [];

parfor i = 1:num
    temp_signal = signal{i};
    temp_prob = result{i};
    Fs = 1000;
    featuresFs = 50;
    [heartRate, systolicTimeInterval] = getHeartRateSchmidt(temp_signal, Fs);
    pi_vector = [0.25,0.25,0.25,0.25];
    [~, ~, qt] = viterbiDecodePCG_Springer(temp_signal,temp_prob, pi_vector, heartRate, systolicTimeInterval, featuresFs,0,alpha);
    preds_news_test{i} = qt;
    right_viterbi_num(i) = sum(qt==a{i});
end
acc_viterbitest_all = sum(right_viterbi_num)/sum(LEN_sound);
[aSe_test,aP_test,aF1_test]=F1_score(preds_news_test,index_use,a);
