%function [Accuracy MatrixProjectionW] = fisherfaceFreeTest( TrainingVector,TestingVector,labelvector_test,NumOfTraining,NumOfClass,NumPerClass,dim,DistMeasure)
function [Accuracy MatrixProjectionW] = fisherfaceFreeTest( TrainingVector,TestingVector,labelvector_train,labelvector_test,NumOfTraining,NumOfClass,dim,DistMeasure)

%create MeanVector
MeanVector=zeros(dim,NumOfClass);
for i=1:NumOfClass
   MeanVector(:,i)=mean(TrainingVector(:,((i-1)*NumOfTraining+1):i*NumOfTraining)')';
end
MeanVectorOfAllVector=mean(MeanVector')';

%create MatrixW
MatrixW=zeros(dim,NumOfClass*NumOfTraining);
for i=1:NumOfClass
   for j=1:NumOfTraining
      MatrixW(:,(i-1)*NumOfTraining+j)=(TrainingVector(:,(i-1)*NumOfTraining+j)-MeanVector(:,i))/sqrt(NumOfTraining*NumOfClass);
   end
end

%create MatrixB
MatrixB=zeros(dim,NumOfClass);
for i=1:NumOfClass
	MatrixB(:,i)=(MeanVector(:,i)-MeanVectorOfAllVector)/sqrt(NumOfClass);
end
TimeForBothDirect=clock;
%create MatrixT
MatrixT=zeros(dim,NumOfClass*NumOfTraining);
for i=1:NumOfClass
   for j=1:NumOfTraining
      MatrixT(:,(i-1)*NumOfTraining+j)=(TrainingVector(:,(i-1)*NumOfTraining+j)-MeanVectorOfAllVector)/sqrt(NumOfTraining*NumOfClass);
   end
end

% MatrixStt=MatrixT'*MatrixT;
% 
% [Ut,St,Vt]=svd(MatrixT);
% [RowOfSt,ColOfSt]=size(St);
% i=1;
%  while((i<=min(RowOfSt,ColOfSt)))
%     if(St(i,i)>0.0000001)
%        i=i+1;
%    else
%        break;
%    end
% end
%  rs=i-1;
%  if rs>(NumOfClass*NumOfTraining-1)
%     rs=(NumOfClass*NumOfTraining-1);
% end
% Temp=Ut(:,1:rs);
% % for i=1:size(Temp,2)
% %     Temp(:,i)=Temp(:,i)/norm(Temp(:,i));
% % end
% 
% 
% % test whether need to descard the first 3 component
% % Y=diag(St);
% % [Y,I] = sort(-Y);
% % Temp=Temp(:,I);
% % % discard3com=3;
[Ut,St] = pcacov(MatrixT*MatrixT');
[RowOfSt,ColOfSt] = size(St);
i = 1;
while(i<=RowOfSt)
    if (St(i) >0.0000001)
        i = i + 1;
    else
        break;
    end
end
rs = i -1;
% if rs>(NumOfClass*NumOfTraining-NumOfClass-1)
%     rs=(NumOfClass*NumOfTraining-NumOfClass-1);
% end
Temp = Ut(:,1:rs);

discard3com=0;

%rs=(NumOfClass*NumOfTraining-NumOfClass);
MatrixY=Temp(:,1+discard3com:rs+discard3com);   % fixed number of the first step of PCA of St

MatrixSwp=(MatrixY'*MatrixW)*(MatrixY'*MatrixW)';
% bInv=0;
% while ~bInv
%     EigSwp=eig(MatrixSwp);
%     if all(EigSwp>1e-5)
%         bInv=1;
%     else
%         rs=rs-1;
%         MatrixY=Temp(:,1:rs);
%         MatrixSwp=(MatrixY'*MatrixW)*(MatrixY'*MatrixW)';
%     end
% end
% rs;
%lda
MatrixSbp=(MatrixY'*MatrixB)*(MatrixY'*MatrixB)';

% [USwp,Swptt,VSwp]=pcacov(MatrixSwp);
% clear temp;
% [Row_swptt,Col_swptt] = size(Swptt);
% 
% i=1;
%  while((i<= Row_swptt))
%     if(Swptt(i)>0.0)
%        i=i+1;
%    else
%        break;
%    end
% end    
% dim_Swppt = i-1;
% for i = 1 : dim_Swppt
%     Temp_Swptt(i,i) = 1/sqrt( Swptt(i) );
% end
% temp = USwp(:,1:dim_Swppt)*Temp_Swptt;
% MatrixSbpp=temp'*MatrixSbp*temp;
% 
% [USbpp,Sbpptt,VSbpp]=pcacov(MatrixSbpp);
% 
% 
% Numwp=min(size(MatrixSwp,1),NumOfClass);
% MatrixA=USbpp(:,1:(NumOfClass-1));

[GeneralizedVec,GeneralizedD] = pcacov(inv(MatrixSwp)*MatrixSbp);
MatrixProjectionW = MatrixY*GeneralizedVec(:,1:(NumOfClass-1));
%MatrixProjectionW=MatrixY*temp*MatrixA;


% %% 1 KNN
% X           = MatrixProjectionW'*TrainingVector;
% Mdl         = fitcknn(X',labelvector_train,'NumNeighbors',NumNeighbors,'Standardize',1);
% 
% Y           = MatrixProjectionW'*TestingVector;
% 
% count       = 0;
% for i = 1:length(labelvector_test)
%     label       = predict(Mdl,Y(i)');
%     
%     if labelvector_test(i) == label
%         count   = count+1;
%     end
% end
% Accuracy = count/length(labelvector_test)






%%
% compute accuracy

[SizeOfSampleVector,colForSampleVector]=size(MatrixProjectionW'*MeanVector(:,1));

SampleVector=zeros(SizeOfSampleVector,NumOfClass);
TempVector=zeros(SizeOfSampleVector,NumOfClass*NumOfTraining);
for i=1:NumOfClass
    for j=1:NumOfTraining
        TempVector(:,(i-1)*NumOfTraining+j)=MatrixProjectionW'*TrainingVector(:,(i-1)*NumOfTraining+j);           
    end
   SampleVector(:,i)=mean(TempVector(:,(i-1)*NumOfTraining+1:i*NumOfTraining)')';
end

%% me
count = 0;
TainingSamplesMean  = SampleVector;
VectorForTest       = MatrixProjectionW'*TestingVector;
if strcmp(DistMeasure, 'mindist')
    for i = 1:length(labelvector_test)
        for k = 1:NumOfClass
            DistanceArray(k)    = norm(TainingSamplesMean(:,k)-VectorForTest(:,i));   
        end
        [M,I]               = min(DistanceArray(:));
        if I == labelvector_test(i)
            count   = count+1;
        end
    end
else
end
Accuracy = count/length(labelvector_test)



% %
% TestingPerClass = NumPerClass - NumOfTraining;	
% DistanceArray=zeros(NumOfClass*NumOfClass,TestingPerClass);
% TempDistance=zeros(NumOfTraining,1);
% 
% for i=1:NumOfClass
%     for j=1:TestingPerClass
%      % j
%         for k=1:NumOfClass
%             VectorForTest=MatrixProjectionW'*TestingVector(:,(i-1)*TestingPerClass+j);
%             if strcmp(DistMeasure, 'mindist')
%                 VectorForSample=SampleVector(:,k);
%                 DistanceArray((i-1)*NumOfClass+k,j)=norm(VectorForSample-VectorForTest);         
%             else
%                 for l=1:NumOfTraining
%                     TempDistance(l,1)=norm(TempVector(:,(k-1)*NumOfTraining+l)-VectorForTest,2);
%                 end           
%                 DistanceArray((i-1)*NumOfClass+k,j)=min(TempDistance);                        
%             end
%         end
%    end
% end
% Accuracy=zeros(1,NumOfClass);
% 
% for i=1:NumOfClass
%     for j=1:TestingPerClass 
%    	    y1=(DistanceArray((i-1)*NumOfClass+1:i*NumOfClass,j))';
%         tvalue=y1(i);
%         y2=sort(y1);
%         index=find(spones(y2-tvalue)-1);
%         for u=index:NumOfClass
%             Accuracy(u)=Accuracy(u)+1;          
%         end  
%     end
% end
% Accuracy=Accuracy/(NumOfClass*TestingPerClass);
% Accuracy(1,1:2)
