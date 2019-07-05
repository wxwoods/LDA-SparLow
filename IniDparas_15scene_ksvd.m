% =========================================================================
% An example code for the algorithm proposed in
%
% Zhuolin Jiang, Zhe Lin, Larry S. Davis. 
% "Label Consistent K-SVD: Learning A Discriminative Dictionary for 
% Recognition". TPAMI, 2013, 35(11): 2651-2664
%
% Author: Zhuolin Jiang (zhuolin@umiacs.umd.edu)
% Date: 12-30-2013
% =========================================================================
function [Dinit,training_feats,testing_feats, labelvector_train, labelvector_test,pars] = IniDparas_15scene_ksvd(param)
% clear all;
% close all;
%clc;
addpath(genpath('./spams-matlab/'));
addpath(genpath('./OMPbox/'));
addpath(genpath('./ksvdbox/'));
%load('./trainingdata/spatialpyramidfeatures4caltech101.mat');
%load('C:\wx\code\5ODL_LDA\call_comparators\incrementallearning_LCKSVD_shared\trainingdata\spatialpyramidfeatures4caltech101.mat');
load('C:\Users\Administrator\Desktop\LDA_SparLow\dataset\spatialpyramidfeatures4scene15.mat')
%% constant
personnumber            = param.personnumber; % person number for evaluation
% constant for incremental dictionary learning
pars.gamma              = 1e-6;
%pars.lambda = 0.5;
pars.lambda             = param.lamda1;
pars.lambda2            = param.lamda2;
pars.mu                 = 0.6; % ||Q-AX||^2
pars.nu1                = 1e-6; % regularization of A
pars.nu2                = 1e-6; % regularization of W
pars.rho                = 10; % initial learning rate
pars.maxIters           = 20; % iteration number for incremental learning
pars.batchSize          = 60;

pars.iterationini       = 100; % iteration number for initialization

pars.initialDic_type    = param.initialDic_type;

pars.ntrainsamples      = param.perclass_train; 
pars.numBases           = personnumber*pars.ntrainsamples; % dictionary size
%pars.numBases          = fix(personnumber*pars.ntrainsamples/2); % dictionary size
pars.numBases           = personnumber*param.number_perclass_dict; % dictionary size
pars.dataset            = 'caltech101';

% constant for LC-KSVD2
sparsitythres           = 40;               % sparsity prior
sqrt_alpha              = 0.0012;           % weights for label constraint term
sqrt_beta               = 0.0012;           % weights for classification err term
iterations              = 50;               % iteration number
iterations4ini          = 20;               % iteration number for initialization
dictsize                = personnumber*30;  % dictionary size

%% get training and testing data
[testing_feats,H_test,training_feats,H_train] = obtaintraingtestingsamples(featureMat,labelMat,pars.ntrainsamples);

para.reducedMethod          = param.reducedMethod;
if strcmpi(para.reducedMethod, 'wavelet')
%     [training_feats]    = data_preprocessing(training_feats);
%     training_feats      = training_feats/max(abs(training_feats(:)));
%     
%     [testing_feats]     = data_preprocessing(testing_feats);
%     testing_feats       = testing_feats/max(abs(testing_feats(:)));
elseif strcmpi(para.reducedMethod, 'PCA')
    options.ReducedDim  = param.ReducedDim_SPM;
%    options.ReducedDim  = 512;
    fea                 = training_feats';
    eigvector           = PCA_cifar(fea, options);
    Data                = fea*eigvector;
    training_feats      = Data'; 
    
    Data                = testing_feats'*eigvector;
    testing_feats       = Data'; 
else
end

if param.mean_flag == 1    
    param.vecOfMeans_test = mean(testing_feats);
    testing_feats = testing_feats-ones(size(testing_feats,1),1)*param.vecOfMeans_test;
    
    param.vecOfMeans_train = mean(training_feats);
    training_feats = training_feats-ones(size(training_feats,1),1)*param.vecOfMeans_train;    
elseif param.mean_flag == 0
    testing_feats = testing_feats'; training_feats = training_feats';
    
    param.vecOfMeans_test = mean(testing_feats);
    testing_feats = testing_feats-ones(size(testing_feats,1),1)*param.vecOfMeans_test;
    
    param.vecOfMeans_train = mean(training_feats);
    training_feats = training_feats-ones(size(training_feats,1),1)*param.vecOfMeans_train; 
    
    testing_feats = testing_feats'; training_feats = training_feats';
end

if param.normalize_flag == 1
   testing_feats    = testing_feats/(diag(sqrt(diag(testing_feats'*testing_feats))));%   
   training_feats    = training_feats/(diag(sqrt(diag(training_feats'*training_feats))));%   
end

%% get the subsets of training data and testing data
% it is related to the variable 'personnumber'
[labelvector_train,~]   = find(H_train);
[labelvector_test,~]    = find(H_test);
trainsampleid           = find(labelvector_train<=personnumber);
testsampleid            = find(labelvector_test<=personnumber);
trainingsubset          = training_feats(:,trainsampleid);
testingsubset           = testing_feats(:,testsampleid);
H_train_subset          = H_train(1:personnumber,trainsampleid);
H_test_subset           = H_test(1:personnumber,testsampleid);

%% Incremental Dictionary learning
% initialization
%[Dinit,Winit,Tinit,Q_train] = paramterinitialization(trainingsubset,H_train_subset, pars);
[Dinit] = paramterinitializationMe(trainingsubset,H_train_subset, pars);
end
% pars.D = Dinit;
% pars.A = Tinit;
% pars.W = Winit;
% 
% fprintf('\nIncremental dictionary learning...');
% [model,stat] = onlineDictionaryLearning(pars, trainingsubset, H_train_subset, Q_train);
% fprintf('done! it took %f seconds',toc);
% for ii=1:pars.maxIters
%     load(fullfile('tmp',sprintf('model_%d_%d_%s.mat',ii,pars.numBases,pars.dataset)),'model');
%     D1 = model.D;
%     W1 = model.W;
%     % classification
%     [~,stat.accuracy(ii)] = classification(D1, W1, testingsubset, H_test_subset, sparsitythres);
%     fprintf('\nFinal recognition rate for OnlineDL is : %f , objective function value: %f', stat.accuracy(ii), stat.fobj_avg(ii));
%     fprintf('%.4f ',stat.accuracy(ii));
%     if ~mod(ii, 10),
%         fprintf('\n');
%     end
% end
% fprintf('Final recognition rate for OnlineDL is : %f\n', max(stat.accuracy(:)));
% % plot the objective function values for all iterations
% figure,
% plot(stat.fobj_avg(:),'marker','o','linewidth',2,'linestyle','--','color','m');
% figure,plot(stat.accuracy(:),'marker','s','linewidth',2,'linestyle','--','color','r');
