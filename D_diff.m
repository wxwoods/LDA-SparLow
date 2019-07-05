function D_egrad = D_diff(X,Phi,B,W,P,param)
%    D           = zeros(size(param.D));
    D           = param.D;
    Dsize       = size(D);
    D_egrad     = zeros(Dsize);
    Phi_size    = size(Phi);
    denominator = trace(Phi*W*Phi'*P);
    nominator   = trace(Phi*B*Phi'*P);
    U           = P*Phi*B;
    V           = P*Phi*W;
    mu          = param.D_thresh;
if strcmpi(param.null_space, 'Skip the null space') 
    coef_null   = param.lambda_fullRank * (1/(Phi_size(2)*log(Phi_size(2)))) * 2/Phi_size(2) ;   
    Z           = Phi * inv( 1/Phi_size(2) * Phi'*Phi );
end
    for i = 1:(Phi_size(2))
        delta_index         = find(Phi(:,i)~=0);%find the index where element is not equal to zero
        delta_index_size    = size(delta_index,1);% find nonzero number
        K_inverse           = InversK(D(:,delta_index),param.lamda2);
        
%         for j = 1:delta_index_size
%             if  Phi(delta_index(j),i)>0
%                 S(j) = 1;
%             else
%                 S(j) = -1;
%             end
%         end
        S  = sign(Phi(delta_index,i));
        S  = S';
        same_bracket = (D(:,delta_index)'*X(:,i)- param.lamda1*S');%(D'X_i-lamda1*S)
        T_U          = same_bracket*U(delta_index,i)' + U(delta_index,i)*same_bracket';
        
        Firstpart    = X(:,i)*U(delta_index,i)'*K_inverse - D(:,delta_index)*K_inverse*T_U*K_inverse;
        
        T_V          = same_bracket*V(delta_index,i)' + V(delta_index,i)*same_bracket';
        
        Secondpart   = X(:,i)*V(delta_index,i)'*K_inverse - D(:,delta_index)*K_inverse*T_V*K_inverse;
        
        D_egrad(:,delta_index) =  D_egrad(:,delta_index) + 2*param.Penalty_sum*(1/denominator * Firstpart - nominator/denominator^2 * Secondpart);
        
%         if strcmpi(param.inco, 'Dinctionary is incoherent') 
%             Gram = D(:,delta_index)'*D(:,delta_index);
%             Gram_sign = sign(Gram);
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
%            end
%             %D_egrad(:,delta_index) = D_egrad(:,delta_index) + param.lambda3*2*D(:,delta_index)*( Gram - I );
%             D_egrad(:,delta_index) = D_egrad(:,delta_index) - param.lambda3*4*D(:,delta_index)*( Gram - I );
%         end
      
    if strcmpi(param.null_space, 'Skip the null space') 
%         coef_null = - param.lambda_fullRank * (1/(Phi_size(2)*log(Phi_size(2)))) * 2/Phi_size(2) ;    
        T_Z         = same_bracket*Z(delta_index,i)' + Z(delta_index,i)*same_bracket';
        nullpart    = X(:,i)*Z(delta_index,i)'*K_inverse - D(:,delta_index)*K_inverse*T_Z*K_inverse;
        D_egrad(:,delta_index) = D_egrad(:,delta_index) + coef_null*nullpart;
        clear nullpart T_Z;
    end
        
        
%       if strcmpi(param.changes_dic, 'Near to original dictionary')     
%          D_egrad(:,delta_index) = D_egrad(:,delta_index) - param.lambda_changes_dic*2 * ( D(:,delta_index) - param.D00(:,delta_index) );   
%       end

        clear same_bracket S K_inverse delta_index T_U T_V Firstpart Secondpart;
    end
      if strcmpi(param.changes_dic, 'Near to original dictionary')     
         D_egrad = D_egrad - param.lambda_changes_dic*2 * ( D - param.D00 );   
      end    
      
%     if strcmpi(param.inco, 'Dinctionary is incoherent') 
%             Gram = D'*D;
%             Gram_sign = sign(Gram);
%             I = eye(Dsize(2));
%             for jj = 1:Dsize(2)
%                 for kk = 1:Dsize(2)
%                     if jj ~= kk
%                         if abs(Gram(jj,kk))<=mu
%                             I(jj,kk) = Gram(jj,kk);
%                         else
%                             I(jj,kk) = Gram_sign(jj,kk)*mu;
%                         end
%                     end
%                 end
%            end
%             %D_egrad(:,delta_index) = D_egrad(:,delta_index) + param.lambda3*2*D(:,delta_index)*( Gram - I );
%             D_egrad = D_egrad - param.IncoDic*4*D*( Gram - I );
%     end
    if strcmpi(param.inco, 'Dinctionary is incoherent') 
       I_D              = eye(size(param.D,2)) + ones(size(param.D,2));
       Mat              = param.D'*param.D;
       D_egrad          = D_egrad  - 4*param.IncoDic*param.Penalty_D*(param.D*(Mat./(I_D-Mat.^2)));
    end
%     if strcmpi(param.inco, 'Dinctionary is incoherent') 
%        I = eye(Dsize(2));
%        D_egrad = D_egrad + param.lambda3*2*D*(D'*D-I);
%     end

%     if strcmpi(param.D_LDA, 'disriminant Dictionary') 
%         denominator = trace(D*param.W_D*D');
%         nominator = trace(D*param.B_D*D');
%         D_egrad = D_egrad + param.lambda4*2*( denominator*D*param.B_D - nominator/denominator^2*D*param.W_D );
%     end
end
function InversValue = InversK(D_delta,lamda2)
    Dsize = size(D_delta);
    identity_matrix = eye(Dsize(2));
    temp = D_delta'*D_delta+lamda2*identity_matrix;
    InversValue = inv(temp);
end
% Dsize = size(D);
% D_update = zeros(Dsize);
% %Phi = param.SamplingMatrix;
% Wsize = size(W);
% tic
% for i = 1:(Wsize(2)-1)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   derivative of W_k
% %% first part
%     %update_D_no = i
%     delta_index1 = find(W(:,i)~=0);%find the index where element is not equal to zero
%     delta_index1_size = size(delta_index1,1);% find nonzero number
%     Pt = InversK(D(:,delta_index1),param.lamda2);
%         %P_tPt = InvK*InvK';
%     Firstpart1 = -X(:,i)*( W(:,i+1) - B(:,delta_index1)*W(delta_index1,i) )'*B(:,delta_index1)*Pt;
%     for j = 1:delta_index1_size
%         if W(delta_index1(j),i)>0
%             S(j) = 1;
%         else
%             S(j) = -1;
%         end
%     end
%      same_bracket = (D(:,delta_index1)'*X(:,i)- param.lamda1*S');%(D'X-ZS)
%      Qt = same_bracket*( W(:,i+1) - B(:,delta_index1)*W(delta_index1,i) )'*B(:,delta_index1);
%     Firstpart =  Firstpart1+D(:,delta_index1)*Pt*(Qt+Qt')*Pt;
%     
%     D_update(:,delta_index1) =  D_update(:,delta_index1) + Firstpart;
%     clear Pt Qt S delta_index1 delta_index1_size same_bracket Firstpart Firstpart1;
% %% second part
% %%K+1  
%     delta_index2 = find(W(:,i+1)~=0);%find the index where element is not equal to zero
%     delta_index2_size = size(delta_index2,1);
%         Pt1 = InversK(D(:,delta_index2),param.lamda2);
%     secondpart1 = X(:,i+1)*( W(delta_index2,i+1) - B(delta_index2,:)*W(:,i) )'*Pt1;
%     for j = 1:delta_index2_size
%         if W(delta_index2(j),i+1)>0
%             S(j) = 1;
%         else
%             S(j) = -1;
%         end
%     end
%     same_bracket2 = (D(:,delta_index2)'*X(:,i+1)- param.lamda1*S');
%     Qt1 = same_bracket2*( - W(delta_index2,i+1) + B(delta_index2,:)*W(:,i) )';
%     
%     Secondpart =  secondpart1 + D(:,delta_index2)*Pt1*( Qt1+Qt1' )*Pt1;
%     
%     D_update(:,delta_index2) =  D_update(:,delta_index2) + Secondpart;
%     clear Pt1 Qt1 S delta_index2 delta_index2_size same_bracket2 Secondpart secondpart1;
% end
%     
%     if param.Inco == 1
%         B_norm = normalize_D(B);
%         D_update = D_update + param.lamda4*(D*B_norm*B_norm');
%     end
% toc
% 
