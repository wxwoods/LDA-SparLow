% (c) Xian Wei, Research Group for Geometric Optimization and Machine Learning
% Muenchen, 2014. Contact: xian.wei@tum.de
%cite：
% @article{wei2019trace,
% title={Trace Quotient with Sparsity Priors for Learning Low Dimensional Image Representations},
% author={Wei, Xian and Shen, Hao and Kleinsteuber, Martin},
% journal={IEEE transactions on pattern analysis and machine intelligence},
% year={2019},
% publisher={IEEE}
% }

close all; clear all;
% Setup;
%param.select_OMP_LASSO          = 1; % omp: 1; lasso: 2;  
param.lamda1                    = 0.05;  
param.lamda2                    = 0.0000001; 
param.mode                      = 2; % elastic net in mexTrainDL;

param.normalize_flag            = 1; % 1 normalize the data
param.mean_flag                 = 1; % 2 ways to center the data, 1 column or 0 row, others means no centering

param.perclass_train            = 100;
param.number_perclass_dict      = 30;
param.reducedMethod             = 'PCA'; % 'wavelet'; 'PCA'; 'Random'
param.initialDic_type           = 'random'; % 'random'; 'DCT'; 'data'
param.ReducedDim_SPM            = 512; % 3000 to 512
param.personnumber              = 15; % 
%Initialize dictionary and split dataset
[Psi,train_prepare,TestingVector, labelvector_train, labelvector_test,pars] = IniDparas_15scene_ksvd(param);
%%
number_perclass_dict = param.number_perclass_dict; number_class = 15;  number_perclass_train = param.perclass_train; number_perclass_test = 0;
train_prepare           = double(train_prepare); 
TestingVector           = double(TestingVector); 

param.classes         = number_class; 
param.labels_train    = labelvector_train;
param.labels_test     = labelvector_test;
param.labels_dic      = labelvector_train;
param.classes         = number_class;
param.Reduced_dims    = param.classes-1;

NumOfTraining = number_perclass_train; NumOfClass = number_class; NumPerClass = number_perclass_train+number_perclass_test;
DistMeasure = 'mindist';
dim_dl = size(train_prepare,1); 
[Accuracy_directLDA, MatrixProjectionW2] = directldaFreeTest( train_prepare,TestingVector,param.labels_train,param.labels_test,NumOfTraining,NumOfClass,dim_dl,DistMeasure);
[Accuracy_fisherface MatrixProjectionW] = fisherfaceFreeTest( train_prepare,TestingVector,param.labels_train,param.labels_test,NumOfTraining,NumOfClass,dim_dl,DistMeasure);
%%
Psi                     = double(Psi); 
Psi_random_mi           = check_incohe(Psi);
param.D_size            = size(Psi);
param.D                 = Psi;
param.labels_dic        = zeros(1,size(Psi,2));

param = init_parameters(param); %%%%%%%%%%%%%%%%%%%

for i = 1:number_class
    param.labels_dic(1, ((i-1)*number_perclass_dict+1) : (i*number_perclass_dict))      = i;
end

TrainingVector = train_prepare; 

param.lamda1       = param.lamda1;    
param.lamda2       = 0.0000001; 
param.paramLasso       = struct(...
            'mode',   2, ...            
            'lambda', param.lamda1, ...
            'lambda2', param.lamda2, ...
            'L',      floor(0.9*param.N*10)  );

train_DL = mexLasso(TrainingVector, Psi, param.paramLasso);
sparsity_DL = length(find(train_DL~=0))/size(TrainingVector,2)

param.lamda1       = param.lamda1;    
param.lamda2        = 0.0000001; 
param.paramLasso_Test      = struct(...
            'mode',   2, ...            
            'lambda', param.lamda1, ...
            'lambda2', param.lamda2, ...
            'L',      floor(0.9*param.N*10)  );
test_DL = mexLasso(TestingVector, Psi, param.paramLasso_Test);
sparsity_DLTest = length(find(test_DL~=0))/size(TestingVector,2)

dim_dl = size(train_DL,1); 
[Accuracy_directLDA, MatrixProjectionW2] = directldaFreeTest( train_DL,test_DL,param.labels_train,param.labels_test,NumOfTraining,NumOfClass,dim_dl,DistMeasure);
[Accuracy_fisherface MatrixProjectionW] = fisherfaceFreeTest( train_DL,test_DL,param.labels_train,param.labels_test,NumOfTraining,NumOfClass,dim_dl,DistMeasure);
%%
param.D00 = param.D;
param.regularizations = 'There many regularizations';
param.inco = 'Dinctionary is incoherent';
param.inco_method  = 'Global incoherent dictionary'; % other is 'Local incoherent dictionary';
param.null_space = 'Don not skip the null space';
param.changes_dic = 'Near to original dictionary';
param.fisher_style = 'Standard_LDA'; % denumerator is S_w,within class scatter matrix
Dsize = size(param.D);
param.D_thresh  = sqrt( (Dsize(2)- Dsize(1))/( Dsize(1)*(Dsize(2)-1) ) );
param.t  = 0;
param.lambda_changes_dic = 5*(1e-2); % 5*(1e-2)
param.IncoDic  = 2.5*1e-4; %2.5*1e-3

param.proj = MatrixProjectionW;
param.P = MatrixProjectionW*MatrixProjectionW';
param.Reduced_dims = size(MatrixProjectionW,2);
param.initial_ortho_projection = MatrixProjectionW;
%%%%%%%%%%%%FUNCTIONS for ODL.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.TestingVector = TestingVector;
param.Reduced_dims = size(param.initial_ortho_projection,2);
param.NumOfTraining = number_perclass_train; param.NumOfClass = number_class; param.NumPerClass = number_perclass_train+number_perclass_test;
%format long;
DistMeasure = 'mindist';
param.DistMeasure = DistMeasure;
param.max_iter           = 3; % Maximal main iterations. 
param.mu_W               = 0.0; % 0.1
param.lambda_changes_dic = 0; % 5*(1e-2)
param.Penalty_sum        = 1;
param.Penalty_D          = 1;

param  = LDA_SparLow(train_prepare, param);



