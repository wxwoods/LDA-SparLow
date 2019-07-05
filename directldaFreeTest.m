%function [Accuracy, MatrixProjectionW] = directldaFreeTest( TrainingVector,TestingVector,labelvector_test,NumOfTraining,NumOfClass,NumPerClass,dim,DistMeasure)
function [Accuracy, MatrixProjectionW] = directldaFreeTest( TrainingVector,TestingVector,labelvector_train,labelvector_test,NumOfTraining,NumOfClass,dim,DistMeasure)

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

MatrixSB=MatrixB'*MatrixB;

[Ub,Sb,Vb]=svd(MatrixSB);
[RowOfSb,ColOfSb]=size(Sb);
i=1;
while((i<=RowOfSb))
   if(Sb(i,i)>1e-7)
      i=i+1;
   else
      break;
   end
end
rs=i-1;
 if rs>(NumOfClass-1)
   rs=(NumOfClass-1);
 end

for i=1:rs
%     MatrixY(:,i)=(MatrixB*Ub(:,i));
    MatrixY(:,i)=(MatrixB*Ub(:,i))/norm(MatrixB*Ub(:,i));
end

MatrixDb=Sb(1:rs,1:rs);
for i=1:rs
    MatrixDb(i,i)=MatrixDb(i,i)^(-1/2);
end

MatrixZ=MatrixY*MatrixDb;
MatrixSwp=MatrixZ'*MatrixW*MatrixW'*MatrixZ;

[Uw,Sw,Vw]=svd(MatrixSwp);
[RowOfSw,ColOfSw]=size(Sw);

MatrixA=Uw(:,diag(Sw)<1);


%MatrixQ=MatrixY*MatrixA;

%MatrixSbp=(MatrixQ'*MatrixB)*(MatrixQ'*MatrixB)';


MatrixProjectionW=MatrixZ*MatrixA;
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












% %%
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
% 
