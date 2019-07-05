% (c) Simon Hawe, Lehrstuhl fuer Datenverarbeitung Technische Universitaet
% Muenchen, 2012. Contact: simon.hawe@tum.de
%% Exponentiall mapping
function [Y,proj] = exp_mapping_Grassm(X, U, t, param)
% D(:,sel) = bsxfun(@times,D(:,sel),cos(t.*Norm_dx(:,sel)))+...
%                bsxfun(@times,dX(:,sel),(sin(t.*Norm_dx(:,sel))./Norm_dx(:,sel)));
% D =  bsxfun(@times,D, 1./sqrt(sum(D.^2))); %Normalization due to numerical things
%% 1 exponetial projection
%  k = 1; % nxpx1
% %         if nargin == 3
% %             tU = t*U;
% %         else
% %             tU = U;
% %         end
%         tU = t*U;
%         Y = zeros(size(X));
%         for i = 1 : k
%             [u s v] = svd(tU(:, :, i), 0);
% %             cos_s = diag(cos(diag(s)));
% %             sin_s = diag(sin(diag(s)));
%             cos_s = diag(cos(diag(s)));
%             sin_s = diag(sin(diag(s)));
%             Y(:, :, i) = X(:, :, i)*v*cos_s*v' + u*sin_s*v';
% %                           % From numerical experiments, it seems necessary to
% %                           % re-orthonormalize. This is overall quite expensive.
% %             [q, unused] = qr(Y(:, :, i), 0); %#ok
% %             Y(:, :, i) = q;
%             [q ~] = uqr(Y(:, :, i));
%             Y(:, :, i) = q*q';
%             %Y(:, :, i) = q(:,1:param.Reduced_dims)*q(:,1:param.Reduced_dims)';
%         end
%        proj = q;
%% 2 retraction
% 
%         Y = X + t*U;
%         for i = 1 : k
%             % We do not need to worry about flipping signs of columns here,
%             % since only the column space is important, not the actual
%             % columns. Compare this with the Stiefel manifold.
%             % [Q, unused] = qr(Y(:, :, i), 0); %#ok
%             % Y(:, :, i) = Q;
%             
%             % Compute the polar factorization of Y = X+tU
%             [u, s, v] = svd(Y(:, :, i), 'econ'); %#ok
%             Y(:, :, i) = u*v';
%         end

%% 3 hao's paper
%          Y = X + t*U;
% %         %tU = t*U;
% %         %U = X + t*U;
%          for i = 1 : k
% %             % We do not need to worry about flipping signs of columns here,
% %             % since only the column space is important, not the actual
% %             % columns. Compare this with the Stiefel manifold.
% %              [u, unused] = uqr(Y(:, :, i));
% 
%               [u, unused] = uqr(Y(:, :, i));
%               Y(:, :, i) = u(:,1:Reduced_dims)*u(:,1:Reduced_dims)';
%               proj = u(:,1:Reduced_dims);
% %             
% % %             % Compute the polar factorization of Y = X+tU
% % %             [u, s, v] = svd(U(:, :, i), 'econ'); %#ok
% % %             Y(:, :, i) = u(:,1:Reduced_dims)*u(:,1:Reduced_dims)';
% % %             proj = u(:,1:Reduced_dims);
%          end
%% 4 % Riemannian exponential mapping;    paper title: The geometry of algorithms with orthogonality constraints
% temp = t*(U*X-X*U);
% side_left = exp(temp);
% side_right = exp(-temp);
% Y = side_left*X*side_right;
% proj = Y;

%% 5. ICASSP 2007
    
    OMG = param.P_skew_egrad;
    [UPT ~] = uqr(eye(size(X,1))+t*OMG);
%    [UPT unused] = qr(eye(size(X,1))-t*OMG);
    Y = UPT*X*UPT';
    rank(Y)
%    proj = (UPT*X)';
    proj = UPT*param.proj;
%     OMG = U*X - X*U;
%    OMG = param.P_skew_egrad;
%     [UPT ~] = uqr(eye(size(X,1))+t*OMG);
%     Y = UPT*X*UPT';
%     rank(Y);
%     proj = Y;

end
					


