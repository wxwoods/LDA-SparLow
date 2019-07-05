function [accuracy_svm, results_SVM_parameters]=CallClassifier_SVM_ODL(param,TEST_ODL)
%clear all;

%close all;
%    load StandardLDA_results_To_UCI_data_inco_1000dic_1000test_2500train2_100loops5_10_3NearDic25_10_2inco_BigSb.mat;
    
 %   load UCI_temp.mat;
 % load Results_data_PIE_data_68c_15dic_40test_100train9618accuracy.mat;
%  load Results_data_PIE_data_68c_20dic_24test_120train9834accuracy.mat;
 
% load results_caltech101_SPM_data_input3060Dic_NormCentered_7575nit.mat
%load results_caltech101_SPM_data512reducedPCA_30input3060Dic_NormCentered_7530nit.mat
%[accuracy] = CallClassifier_SVM_ODL(param,TEST_ODL);

    TrainingVector_ODL2         = param.proj'*param.Phi;
%    TEST_ODL2               = param.proj'*TEST_ODL;
    
    TEST_ODL2                   = TEST_ODL;
     
    training_instance_matrix    = TrainingVector_ODL2';
    testing_instance_matrix     = TEST_ODL2';
   
    if size(param.labels_train,1) == 1
        training_label_vector   = param.labels_train';
        testing_label_vector    = param.labels_test';
    else
        training_label_vector   = param.labels_train;
        testing_label_vector    = param.labels_test;
    end
%     nClass = length(unique(training_label_vector));
%     sepa_train = size(training_label_vector,1)/nClass;
%     sepa_test = size(testing_label_vector,1)/nClass;
%     
%     training_label_vector(sepa_train+1:end,:) = -1;
%     testing_label_vector(sepa_test+1:end,:) = -1;

%% standard classifier
%    accuracy_svm = call_SVM320(training_label_vector,training_instance_matrix,testing_label_vector,testing_instance_matrix)
%% one to one classifier
    [accuracy_svm, results_SVM_parameters] = My_SVM1to1(training_label_vector,training_instance_matrix,testing_label_vector,testing_instance_matrix);
%% one to all classfier
%    [accuracy_svm] = My_SVM1toAll(training_label_vector,training_instance_matrix,testing_label_vector,testing_instance_matrix)

% [accuracy_svm, results_SVM_parameters] = My_SVM1toAll(training_label_vector,training_instance_matrix,testing_label_vector,testing_instance_matrix)
    
    %[result] = multisvm(training_instance_matrix,training_label_vector,TEST_ODL2)
    
    
end