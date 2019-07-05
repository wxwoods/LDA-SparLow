% (c) Xian Wei, Research Group for Geometric Optimization and Machine Learning
% Muenchen, 2014. Contact: xian.wei@tum.de
function [P,Phi,param,f0] = Update_P_SGodl(X,Phi,B,W,D,P,P_grad,param,f0)
Phi = full(Phi); 
Phi_SIZE = size(Phi);
% Phi1 = Phi(:,1:(Phi_SIZE(2)-1));
% Phi2 = Phi(:,2:Phi_SIZE(2));
val = 0;
%% main cost function
alpha 		= 1e-2; %\in(0,0.5)
iter_D      = 0;
max_iter_D  = 150;

%threshold = 1e-4;
% dx         = -D_grad;%approximate optimal point along a way looks like "Z".
 dx_P       = -P_grad;%approximate optimal point along a way looks like "Z".
 
%val         = -D_grad(:)'*D_grad(:);
val         = val-P_grad(:)'*P_grad(:);
%val = -sqrt(-val);
f_c = f0;    

if (param.it == 1) && (param.t == 0)
%    f0          = trace(Phi*B*Phi'*P)/trace(Phi*W*Phi'*P);  
%    t_init = 0;  
%   dx = normalize_D(dx);
%     t_init  = sqrt(sum(dx(:).^2));
%     t = 1/t_init;
%     t_init_B  = sqrt(sum(dx_P(:).^2));
%     t3 = 1/t_init_B;
%     t = max(t,t3);
%     t = sqrt(t);
     t = 1/norm(dx_P(:));
 else
     t = param.t_p;
 end

P0 = P;
D0 = D;
%beta  		= [0.9,0.5,0.3,0.1,0.0333];%is lamda; the most impot=rtant; speed of reduction for step t. should be lager for D
beta  		= [0.9,0.8,0.5,0.5,0.5];%is lamda; the most impot=rtant; speed of reduction for step t. should be lager for D
Norm_dx_P     = sqrt(sum(dx_P.^2));  
sel_dx_P         = Norm_dx_P > 0;

% if strcmpi(param.grassmann.geodesic, 'Martin')
%     if param.it == 1
%        % param.initial_ortho_projection = Q;   % X = Q*Q';
%        param.ortho_projection   = param.initial_ortho_projection; % P = u0*u0';
%     end
%     %Omega              = X*param.P_eGrad - param.P_eGrad*X; % skew hermitian matrix; 
%     Omega              =  param.P_skew_egrad;
% %     G                  = X*Omega - Omega*X; % gradient in Riemannian space; 
% %     H                  = -G; % Direction
%     Deta               = triu(Omega*Omega); %get upper triangular part of matrix
%     gamma_first_derivative      = real(trace(P*param.P_eGrad*Omega));
%     gamma_second_derivative     = real(trace(Deta*P*param.P_eGrad)) - real(trace(Omega*P*Omega*param.P_eGrad));
%     lambda                      = - gamma_first_derivative/gamma_second_derivative;
%     %     % Update
%     I                           = eye(size(Omega,1));
%     [Q, unused]                 = qr(I+lambda*Omega);
%     param.ortho_projection      = Q*param.ortho_projection; %% P = u*u';
%     
%     Y                           = param.ortho_projection * param.ortho_projection';
% end

while (iter_D == 0)||(f0 >= f_c + alpha * t * val) &&(iter_D < max_iter_D)
     display1 = f_c + alpha * t * val;
%     display2 = elastic_net_coss_0 + alpha * t * val_ELA 
    temp = abs(f0-display1);
%     if temp<threshold
%         break;
%     end
    if ((temp/abs(display1))>1000)
        %||((elastic_net_coss/display2)>1000)
     t  = t*beta(3); 
    elseif iter_D == 0
     t  = t;
    elseif((temp/abs(display1))>10)
        %||((elastic_net_coss/display2)>10)
     t  = t*beta(2);
    elseif((temp/abs(display1))>0.1)
        %||((elastic_net_coss/display2)>10)
     t  = t*beta(1);

%     elseif iter_D == 1
%         t  = t;
%    elseif  iter_D>4
    else
     t  = t*beta(1);
    end
    %% Update D and P
%      D  = exp_mapping_sphere(D0, dx, t, Norm_dx_P,sel_dx_P);
%      param.D        = D;
% %      D  = exp_mapping_sphere(D0, dx, t, Norm_dx,sel);
% %      D  = D + t*dx;
% %      D =  bsxfun(@times,D, 1./sqrt(sum(D.^2))); %Normalization due to numerical things
     [P,proj]        = exp_mapping_Grassm(P0, dx_P, t,param);
      param.P        = P;
%     if strcmpi(param.grassmann.geodesic, 'Martin')
%         %P = Y;
%         [P,param] = update_Grassm(P0, dx_P, t, param);
%     else
%         [P,param] = update_Grassm(P0, dx_P, t, param);
%     end
     
%      D =  D0 + t*dx;
%      B = B0 + t*dx_B;    
%       D  = normalize_D(D);
% %P = P0;
%       if strcmpi(param.SR_update, 'Closed form')
%          Phi = elastic_net(X,Phi,D,param); % it will be fast
%       else
%          Phi = mexLasso(X, D, param.paramLasso);
%          Phi = full(Phi); 
%          param.Phi = Phi;
%       end
      
     if strcmpi(param.regularizations, 'There many regularizations') 
            f0          = FuncValue(param);
        else
            f0          = -trace(Phi*B*Phi'*P)/trace(Phi*W*Phi'*P);
      end
    %  f0          = -trace(Phi*B*Phi'*P)/trace(Phi*W*Phi'*P);
     %[elastic_net_coss] = getlossfucfromelasticnet(X,W,D,param);
     
    % f0 = sum(temps(:));
     iter_D = iter_D+1;
     
      if param.verbose
            sparsity = length(find(Phi~=0))/size(Phi,2)
            fprintf('Objective %e ~ current %e ~ stepsize %e \n', f0, f_c, t)
      end    

end
      param.proj     = proj;
      diff = norm(param.proj*param.proj' - P)
 t = t/(beta(1)^2);
% t = t/(beta(1));
param.t_p = t;
estimate_Drho = 0;
end
 
%  function [elastic_net_coss] = getlossfucfromelasticnet(X,W,D,param)
%     elastic_net_coss = 0;
%     temp_sum = (X - D*W).^2;  
%     elastic_net_coss = 0.5*sum(temp_sum(:));
%     Sp_type  = 'PNormAbs';
%     q = 1;
%     mu       = 0;    % Multiplier in log(1+mu*x^2)
%     %[fsp,q_w] = Sparsifying_functions(Sp_type, 'Evaluate', double(W), q, mu);
%     fsp = sum(abs(W(:)));
%     elastic_net_coss = elastic_net_coss + param.lamda1*fsp + 0.5*param.lamda2*sum(W(:).^2);
%  end
%  
%  function [diff_d_ela] = diff_D_elastic(X,W,D)
%     diff_d_ela = D*W*W'-X*W';
%  end
 
 %B = B - Brho*B_update;
 
 %temp2 = norm((W2-(B) * W1).^2);

%%1st loop
% Low_range = estimate_Brho/10;
% Upper_range = estimate_Brho*10;

