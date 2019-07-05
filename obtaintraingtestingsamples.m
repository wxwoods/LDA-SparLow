function [testPart,HtestPart,trainPart,HtrainPart] = obtaintraingtestingsamples(featureMatrix,labelMatrix,numPerClass)
% obtain training and testing features by random sampling
% Inputs
%     featureMatrix      - input features 
%     labelMatrix        - label matrix for input features
%     numPerClass        - number of training samples from each category
% Outputs
%     testPart           - testing features
%     HtestPart          - label matrix for testing features
%     trainPart          - training features
%     HtrainPart         - label matrix for training features
%

numClass = size(labelMatrix,1); % number of objects
testPart = [];
HtestPart = [];
trainPart = [];
HtrainPart = [];

for classid=1:numClass
    col_ids = find(labelMatrix(classid,:)==1);
    data_ids = find(colnorms_squared_new(featureMatrix(:,col_ids)) > 1e-6);   % ensure no zero data elements
    perm = randperm(length(data_ids));
    %perm = [1:length(data_ids)];
    
    trainids = col_ids(data_ids(perm(1:numPerClass)));
    testids = setdiff(col_ids,trainids);
    
    testPart = [testPart featureMatrix(:,testids)];
    HtestPart = [HtestPart labelMatrix(:,testids)];
    trainPart = [trainPart featureMatrix(:,trainids)];
    HtrainPart = [HtrainPart labelMatrix(:,trainids)];
end
