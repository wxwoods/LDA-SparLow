function [Dinit] = paramterinitializationMe(training_feats,H_train,para)
% paramter initialization for incremental dictionary learning

dictsize = para.numBases;
iterations = para.iterationini;
numClass = size(H_train,1); % number of objects
Dinit = []; % for C-Ksvd and D-Ksvd
dictLabel = [];
numPerClass = dictsize/numClass;
for classid=1:numClass
    labelvector = zeros(numClass,1);
    labelvector(classid) = 1;
    dictLabel = [dictLabel repmat(labelvector,1,numPerClass)];
end

for classid=1:numClass
    col_ids = find(H_train(classid,:)==1);
    data_ids = find(colnorms_squared_new(training_feats(:,col_ids)) > 1e-6);   % ensure no zero data elements are chosen
    perm = randperm(length(data_ids));
    %perm = [1:length(data_ids)]; 
    %%%  Initilization for LC-KSVD (perform KSVD in each class)
    Dpart = training_feats(:,col_ids(data_ids(perm(1:numPerClass))));
    param1.mode = 2;
    param1.K = para.numBases;
    param1.lambda = para.lambda;
    param1.lambda2 = para.lambda2;
    param1.iter = iterations;
    if strcmpi(para.initialDic_type, 'Random')
       [row, col]   = size(Dpart);
       [Dpart]      = initialDicRandom(row,col);
       param1.D     = Dpart;
    else
        param1.D = Dpart;
    end
    
    Dpart           = mexTrainDL(training_feats(:,col_ids(data_ids)),param1);
    Dinit           = [Dinit Dpart];
    labelvector = zeros(numClass,1);
    labelvector(classid) = 1;
    dictLabel = [dictLabel repmat(labelvector,1,numPerClass)];
end

% param1.mode = 2;
% param1.K = para.numBases;
% param1.lambda = para.lambda;
% param1.lambda2 = 0;
% param1.iter = iterations;
% 
% param1.D = Dinit;
% 
% Dinit = mexTrainDL(training_feats,param1);

end

function [Dint] = initialDicRandom(row,col)
    M       = col; % one of Second of original data 
    N       = row; %1024
    Phi     = randn(N,M);
    Dint    = Phi./repmat(sqrt(sum(Phi.^2,1)),[N,1]);	
end
% param2.lambda = para.lambda;
% param2.lambda2 = 0;
% param2.mode = 2;
% Xinit=mexLasso(training_feats,Dinit,param2);
% 
% % learning linear classifier parameters
% Winit = inv(Xinit*Xinit'+eye(size(Xinit*Xinit')))*Xinit*H_train';
% Winit = Winit';
% 
% Q = zeros(dictsize,size(training_feats,2)); % energy matrix
% for frameid=1:size(training_feats,2)
%     label_training = H_train(:,frameid);
%     [maxv1,maxid1] = max(label_training);
%     for itemid=1:size(Dinit,2)
%         label_item = dictLabel(:,itemid);
%         [maxv2,maxid2] = max(label_item);
%         if(maxid1==maxid2)
%             Q(itemid,frameid) = 1;
%         else
%             Q(itemid,frameid) = 0;
%         end
%     end
% end
% 
% Tinit = inv(Xinit*Xinit'+eye(size(Xinit*Xinit')))*Xinit*Q';
% Tinit = Tinit';
