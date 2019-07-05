function [f]=FuncValue(param)
%FUNCTION THAT CALCULATES SCATTER MATRIX:
%   B:BETWEEN  CLASS SCATTER MATRIX
%   W:WITHIN CLASS SCATTER MATRIX
%
f               = -param.Penalty_sum*trace(param.Phi*param.B*param.Phi'*param.P)/trace(param.Phi*param.W*param.Phi'*param.P);
trace_Quotient_error    = f;
% if strcmpi(param.inco, 'Dinctionary is incoherent') 
%             Dsize = size(param.D);
%             Gram = param.D'*param.D;
%             Gram_sign = sign(Gram);
%             delta_index_size = Dsize(2);
% %             I = eye(delta_index_size);
%             mu          = param.D_thresh;
%             
%             I = eye(delta_index_size);
%             for jj = 1:delta_index_size
%                 for kk = 1:delta_index_size
%                     if jj ~= kk
%                         if abs(Gram(jj,kk))<=mu
%                             I(jj,kk) = Gram(jj,kk);
%                         else
%                             I(jj,kk) = Gram_sign(jj,kk)*mu;
%                         end
%                     end
%                 end
%             end
%         temp_incoDic = 0.5*param.IncoDic*trace((Gram - I)*(Gram - I)')
%         f = f + temp_incoDic;
% end
%I                   = eye( size(param.D,2) );
temp_incoDic = 0;
I                 = eye(size(param.D,2)) + ones(size(param.D,2));
if strcmpi(param.inco, 'Dinctionary is incoherent') 
    if strcmpi(param.inco_method, 'Global incoherent dictionary') 
        temp_incoDic   =  - param.IncoDic*param.Penalty_D*sum(sum(log(I-(param.D'*param.D).^2)));
%        temp_incoDic   =  temp_incoDic + param.IncoDic_P*param.Penalty_DP*sum(sum(log(I-(param.P'*param.P).^2)));
        f              =  f + temp_incoDic;
    elseif strcmpi(param.inco_method, 'Local incoherent dictionary') 
    else
        temp_incoDic = 0;
    end
else
    temp_incoDic = 0;
end



if strcmpi(param.null_space, 'Skip the null space') 
    [row,col] = size(param.Phi);
    temp = log(det( 1/col*param.Phi'*param.Phi ));
    
   f = f - param.lambda_fullRank * (1/(col*log(col))) * temp;
   clear temp;
end

temp_nearDic = 0;
if strcmpi(param.changes_dic, 'Near to original dictionary') 
%     [row,col] = size(param.Phi);
%     for i = 1:param.classes
%        param.D = param.D0();
%     end    
%    temp = log(det( 1/col*param.Phi'*param.Phi ));     
   temp_nearDic = param.lambda_changes_dic * trace(  (param.D - param.D00)*(param.D - param.D00)' );
    f = f + temp_nearDic;
%    clear temp;
end

% if strcmpi(param.D_LDA, 'disriminant Dictionary') 
%     f = f - param.lambda4*trace(param.D*param.B_D*param.D')/trace(param.D*param.W_D*param.D');
% end

%     [~, l]=size(data);         %CALCULATE SIZE OF DATA
%     clases=unique(Y);          %GET VECTOR OF CLASSES
%     tot_clases=length(clases); %HOW MANY CLASSES
%     B=zeros(l,l);              %INIT B AND W
%     W=zeros(l,l);           
%     overallmean=mean(data);    %MEAN OVER ALL DATA
%     for i=1:tot_clases
%         clasei = find(Y==clases(i)); %GET DATA FOR EACH CLASS
%         xi=data(clasei,:);
%         
%         mci=mean(xi);                       %MEAN PER CLASS
%         xi=xi-repmat(mci,length(clasei),1); %Xi-MeanXi
%         W=W+xi'*xi;                         %CALCULATE W
%         B=B+length(clasei)*(mci-overallmean)'*(mci-overallmean); %CALCULATE B
%     end
fprintf('Total value %e ~ Trace Quotient error %e ~ \n ~ Incoherent part %e ~ Near to initial %e \n', f, trace_Quotient_error, temp_incoDic, temp_nearDic);
end
