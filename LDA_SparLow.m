% Function finds the best dictionary D and projection matrix P that lies on the oblique
% manifold and grassmannian manifold via a steepest gradient method. The cost function to
% be minimized is f(D,P) = trace(S_b(D,X)*P)/trace(S_w(D,X)*P), where S_b
% and S_w are the scatter matrices from fisher criterion.
% (c) Xian Wei, Research Group for Geometric Optimization and Machine
% Learning, TUM.
% Muenchen, 2014. Contact: xian.wei@tum.de
function param = LDA_SparLow(X, param)
% initialize parameters
mu_W                    =  param.mu_W;
param.mu_W              =  mu_W;
D                       =  bsxfun(@times,param.D, 1./sqrt(sum(param.D.^2))); %Normalization due to numerical things
param.D00               =  D;
param.D                 =  D;
P                       =  param.P;
sparsity                =  zeros(param.max_iter+2,1);     % check the sparsity of W.
Error_Rec_DL            =  ones(param.max_iter+2,1);      % error for \| X - D*Phi\|_F^2
number_class            =  param.classes;
number_perclass_train   =  param.perclass_train;
param.t_p = 0;
param.t_D = 0;

%%% 
Total_trainingNumber    = length(param.labels_train);
W = block_centerring_matrix(number_class, number_perclass_train);
%B = (param.perclass_train)*block_betweenin_matrix(number_class, number_perclass_train);
B = Total_trainingNumber/number_class * block_betweenin_matrix(number_class, number_perclass_train);
W_D = block_centerring_matrix(number_class, param.number_perclass_dict);
if mu_W ~= 0
    W_D = W_D + mu_W*eye(size(W_D));
    W   = W + mu_W*eye(size(W));
end

B_D = (param.perclass_train)*block_betweenin_matrix(number_class, param.number_perclass_dict);
%B_D = block_betweenin_matrix(number_class, param.number_perclass_dict);
if strcmpi(param.fisher_style, 'Direct_LDA')
     W   = block_centerring_matrix(1, number_perclass_train*number_class); % denumerator is S_t, global scatter matrix
     W_D = block_centerring_matrix(1, param.number_perclass_dict*number_class);
end

param.B   = B;
param.W   = W;
param.B_D   = B_D;
param.W_D   = W_D;
dictionary_mi = zeros(1,param.max_iter+1);
accuracy_fisher = zeros(1,param.max_iter+1);
accuracy_direct_LDA = zeros(1,param.max_iter+1);
accuracy_huang = zeros(1,param.max_iter+1);
accuracy_standard_LDA = zeros(1,param.max_iter+1);
accuracy_standard_LDA_SVM = zeros(1,param.max_iter+1);
Accuracy_huangODL = 0;
accuracy_stanldaODL = 0;

Func_value = zeros(1,param.max_iter+1);
discriminant_Dic = zeros(1,param.max_iter+1);
param.numerator = zeros(1,param.max_iter+1);
param.denumerator = zeros(1,param.max_iter+1);
param.Gramvalue_dic = zeros(1,param.max_iter+1);

param.dictionary = cell(1,param.max_iter+1);
for k = 1:param.max_iter
    fprintf('---  iteration %4i, learning with data set  ',k);
    if k == 1
        Phi = mexLasso(X, param.D, param.paramLasso);
        %Phi = mexOMP(X, para.D, param.paramOMP);
        Phi = full(Phi);         
        param.Phi = Phi;
        temp_sum = (X - D*Phi).^2;  
        Error_Rec_DL(1) = sum(temp_sum(:)); %norm2 of error, L2 fit for residence : |X-DPhi|_2
        sparsity(1) = length(find(Phi~=0));
        if strcmpi(param.regularizations, 'There many regularizations') 
            f0          = FuncValue(param);
        else            
            f0          = -trace(Phi*B*Phi'*P)/trace(Phi*W*Phi'*P);
        end
        
        [SB SW ST]=scattermat(Phi',param.labels_train');

%         f_compare          = -trace(SB*P)/trace(SW*P);
         if strcmpi(param.fisher_style, 'Direct_LDA')
                f_compare          = -trace(SB*P)/trace(ST*P);
         else
                f_compare          = -trace(SB*P)/trace(SW*P);
         end

%            TrainingVector_ODL = param.P*param.Phi;
         TrainingVector_ODL = param.proj'*param.Phi;
            
         dim_odl = size(TrainingVector_ODL,1); 
         TEST_ODL = mexLasso(param.TestingVector, param.D, param.paramLasso);
            %Phi = mexOMP(X, para.D, param.paramOMP);
         TEST_ODL = full(TEST_ODL);  
%            TEST_ODL = param.P*TEST_ODL;
         TEST_ODL = param.proj'*TEST_ODL;
         [Accuracy_direct_lda_ODL, MatrixProjectionW] = directldaFreeTest( TrainingVector_ODL,TEST_ODL,param.labels_train,param.labels_test,param.NumOfTraining,param.NumOfClass,dim_odl,param.DistMeasure);
         [Accuracy_fisherlda_ODL MatrixProjectionW] = fisherfaceFreeTest( TrainingVector_ODL,TEST_ODL,param.labels_train,param.labels_test,param.NumOfTraining,param.NumOfClass,dim_odl,param.DistMeasure);

         dictionary_mi(1)            = check_incohe(param.D);
            
         accuracy_fisher(1)          = Accuracy_fisherlda_ODL(1);
         accuracy_direct_LDA(1)      = Accuracy_direct_lda_ODL(1);
         accuracy_huang(1)           = Accuracy_huangODL(1);
         accuracy_standard_LDA(1)    = accuracy_stanldaODL(1);
            
        % [accuracy_standard_LDA_SVM(1), results_SVM_parameters]        = CallClassifier_SVM_ODL(param,TEST_ODL);
         Func_value(1)               = f0;
         discriminant_Dic(1)         = trace(param.D*param.B_D*param.D')/trace(param.D*param.W_D*param.D');
            
         param.numerator(1)          = trace(Phi*B*Phi'*P);
         param.denumerator(1)        = trace(Phi*W*Phi'*P);
            
         Gram                        = param.D'*param.D;
         I_D                         = eye(size(Gram,2));
         Gram                        = abs(Gram - I_D);
         param.Gramvalue_dic(1)      = sum(Gram(:));
            
         param.dictionary{1, 1}      = param.D;
    end
    if k==1
        if param.verbose
            Sp_input = f0;
        end
    end
%%  separablly
     %Note that, here "D_egrad" is real symmetric matrix
    param.SR_update         = 'Learning via lasso';
    param.it                = k;  
        
     % 2 udatate P     
     P_before        = param.P;
     D_before        = param.D;
     
    P_egrad     = -P_diff(Phi,B,W,P,param);   % gradient in Euclidean space 
    P_skew      = P*P_egrad - P_egrad*P; % skew hermitian matrix; 
%     %Note that, here "D_grad" is gotten using double Lie bracket
    P_grad      = P*P_skew - P_skew*P; % gradient in Riemannian space; 
    param.it            = k;
    param.P_eGrad       = P_egrad;
    param.P_skew_egrad  = P_skew;
    param.SR_update = 'Learning via lasso';
 %   param.P_eGrad           = P_egrad;
    if strcmpi(param.GlobalMethod, 'Steepes_gradient')
        [P_c,Phi,param,f0] = Update_P_SGodl(X,Phi,B,W,D,P,P_grad,param,f0); % together update; line search.
    elseif strcmpi(param.GlobalMethod, 'CGA')
        [P_c,Phi,param,f0] = Update_P_CG(X,Phi,D,P,P_grad,param,f0); % together update; line search.
    end
    param.P         = P_c; 
    P               = param.P;
 
     % 1 udatate D       
    D_egrad                 = -D_diff(X,Phi,B,W,P,param);% gradient in Euclidean space; 
    D_grad                  = D_egrad - D*diag(diag(D'*D_egrad));
    if strcmpi(param.GlobalMethod, 'Steepes_gradient')
        [D_c,Phi,param,f0]  = Update_D_SGodl(X,Phi,B,W,D,D_grad,P,param,f0); % together update; line search.
    elseif strcmpi(param.GlobalMethod, 'CGA')
        [D_c,Phi,param,f0] = Update_D_CG(X,Phi,D,D_grad,P,param,f0); % together update; line search.
    end
    param.D   = D_c;
    D         = D_c;  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if param.verbose 
        fprintf('Change of the operator: %f\n',norm(D_before(:)-D_c(:)))
        fprintf('Change of the operator: %f\n',norm(P_before(:)-P_c(:)))
        fprintf('Current Fieldy %e ~ Input Fieldy %e\n', f0, Sp_input)
        fprintf('step size: %f\n',param.t)
        %fprintf('Number of Linesearchsteps %d ~ Stepsize: %e ~estimate %e ~ Condition %f\n',ls_iter,t_prev,t_initial,cond(Omega_c))
    end

    param.Phi = Phi;

    if strcmpi(param.SR_update, 'Closed form')
        Phi = mexLasso(X, D, param.paramLasso);
        Phi = full(Phi); 
        param.Phi = Phi;
    end
    P = P_c;
    D = D_c;
    
    param.D   = D;
    param.P   = P;

   TrainingVector_ODL = param.proj'*param.Phi;
   
   dim_odl = size(TrainingVector_ODL,1); 
   TEST_ODL = mexLasso(param.TestingVector, param.D, param.paramLasso);
   %Phi = mexOMP(X, para.D, param.paramOMP);
   TEST_ODL = full(TEST_ODL);  
%         TEST_ODL = param.P*TEST_ODL;
   TEST_ODL = param.proj'*TEST_ODL;
   [Accuracy_direct_lda_ODL, MatrixProjectionW] = directldaFreeTest( TrainingVector_ODL,TEST_ODL,param.labels_train,param.labels_test,param.NumOfTraining,param.NumOfClass,dim_odl,param.DistMeasure);
   [Accuracy_fisherlda_ODL MatrixProjectionW] = fisherfaceFreeTest( TrainingVector_ODL,TEST_ODL,param.labels_train,param.labels_test,param.NumOfTraining,param.NumOfClass,dim_odl,param.DistMeasure);
           
   dictionary_mi(k+1)            = check_incohe(param.D);
            
    accuracy_fisher(k+1)      = Accuracy_fisherlda_ODL(1);
    accuracy_direct_LDA(k+1)      = Accuracy_direct_lda_ODL(1);
    accuracy_huang(k+1)           = Accuracy_huangODL(1);
    accuracy_standard_LDA(k+1)    = accuracy_stanldaODL(1);
    %[accuracy_standard_LDA_SVM(k+1), results_SVM_parameters]        = CallClassifier_SVM_ODL(param,TEST_ODL);
    
    Func_value(k+1)             = f0;
    discriminant_Dic(k+1)       = trace(param.D*param.B_D*param.D')/trace(param.D*param.W_D*param.D');
    
    param.numerator(k+1)        = trace(param.Phi*B*param.Phi'*param.P);
    param.denumerator(k+1)      = trace(param.Phi*W*param.Phi'*param.P);
    
    Gram                        = param.D'*param.D;
    I_D                         = eye(size(Gram,2));
    Gram                        = abs(Gram - I_D);
    param.Gramvalue_dic(k+1)    = sum(Gram(:));
    
    temp_accuracy = max(Accuracy_direct_lda_ODL, Accuracy_fisherlda_ODL);
    if  temp_accuracy > 0.999
         fprintf('************stop while achiving the goal **********\n');
         break;
    end
     k
    
     param.dictionary{1, k+1} = param.D;


    temp_sum            = (X - D*param.Phi).^2;  
    Error_Rec_DL(k+1)   = sum(temp_sum(:)); %norm2 of error, L2 fit for residence : |X-DPhi|_2
    sparsity(k+1)       = length(find(param.Phi~=0));
end
    param.dictionary_mi         = dictionary_mi;
    param.accuracy_fisher       = accuracy_fisher;
    param.accuracy_direct_LDA   = accuracy_direct_LDA;
    param.Func_value            = Func_value;
    param.discriminant_Dic      = discriminant_Dic;
   % param.accuracy_standard_LDA_SVM      = accuracy_standard_LDA_SVM;   
    param.Error_Rec_DL          = Error_Rec_DL;
    param.sparsity              = sparsity;    
    param.TEST_ODL              = TEST_ODL;
    
    save results_saved_data_temp;    
end





