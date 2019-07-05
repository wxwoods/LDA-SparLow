% (c) Xian Wei, Research Group for Geometric Optimization and Machine Learning
% Muenchen, 2014. Contact: xian.wei@tum.de
function diff_result = P_diff(Phi,B,W,P,param)
   denominator  = trace(Phi*W*Phi'*P);
   nominator    = trace(Phi*B*Phi'*P);
   diff_result  = param.Penalty_sum*1/denominator*( Phi*B'*Phi' -  nominator/denominator * Phi*W'*Phi');
end
% Bsize = size(B);%dim(B)=dim(W*W')
% Wsize = size(W);
% diff_result = zeros( Bsize(1),Bsize(2) );
% 
% %diff_result = B;
% % for i = 2:Wsize(2)
% %     diff_result = diff_result + ( B*W(:,i-1)-W(:,i) ) *W(:,i-1)';
% % end
% diff_result =  (B*W(:,1:Wsize(2)-1)-W(:,2:Wsize(2)))*W(:,1:Wsize(2)-1)' + param.lamda3*B;
% %diff_result = diff_result+B;
% if param.Inco == 1
%     B_norm = normalize_D(B);
%     diff_result = diff_result + param.lamda4*(D'*D*B_norm);
% end
