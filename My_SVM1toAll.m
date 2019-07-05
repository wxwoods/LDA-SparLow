% Matlab build-in SVM version_1
function [accuracy, results_SVM_parameters] = My_SVM1toAll(training_label_vector,training_instance_matrix,testing_label_vector,testing_instance_matrix)
% %load fisheriris;
% [~,~,labels] = unique(species);   %# labels: 1/2/3
% data = zscore(meas);              %# scale features
% numInst = size(data,1);
u=unique(training_label_vector);
numLabels=length(u);
%numLabels = max(training_label_vector); % classes
% %# split training/testing
% idx = randperm(numInst);
% numTrain = 100; numTest = numInst - numTrain;
% trainData = data(idx(1:numTrain),:);  testData = data(idx(numTrain+1:end),:);
% trainLabel = labels(idx(1:numTrain)); testLabel = labels(idx(numTrain+1:end));
trainData = training_instance_matrix;  
testData = testing_instance_matrix;
trainLabel = training_label_vector;     
testLabel = testing_label_vector;

numTest = size(testLabel, 1);
numTrain = size(trainLabel, 1);
%# train one-against-all models
model = cell(numLabels,1);
for k=1:numLabels
%    model{k}                = libsvmtrain(double(trainLabel==k), trainData, '-c 1 -g 0.2 -b 1');
Group_all = double(trainLabel==k);
    model{k}                = svmtrain(Group_all, trainData, '-c 1 -g 0.2 -b 1');
%model{k}                = svmtrain(trainData, Group_all);
end

%# Get probability estimates of test instances using each model
prob                    = zeros(numTest,numLabels);
for k=1:numLabels
    [~,~,p]                 = svmpredict(double(testLabel==k), testData, model{k}, '-b 1');
%    [~,~,p]                 = svmpredict(double(testLabel==k), testData, model{k}, '-b 1');
    prob(:,k)               = p(:,model{k}.Label==1);    % Probability of class==k
end

% Predict the class with the highest probability
[~,pred]                = max(prob,[],2);
acc                     = sum(pred == testLabel) ./ numel(testLabel);    % Accuracy
C                       = confusionmat(testLabel, pred);                 % Confusion matrix

accuracy = acc;

results_SVM_parameters.pred = pred;
results_SVM_parameters.confusionMatrix = C;
end