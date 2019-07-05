% (c) Simon Hawe, Lehrstuhl fuer Datenverarbeitung Technische Universitaet
% Muenchen, 2012. Contact: simon.hawe@tum.de
%% Exponentiall mapping
function D = exp_mapping_sphere(D, dX, t, Norm_dx,sel)
D(:,sel) = bsxfun(@times,D(:,sel),cos(t.*Norm_dx(:,sel)))+...
               bsxfun(@times,dX(:,sel),(sin(t.*Norm_dx(:,sel))./Norm_dx(:,sel)));
D =  bsxfun(@times,D, 1./sqrt(sum(D.^2))); %Normalization due to numerical things
					
% D = D + t*dX;
% %D = D*diag(1./sqrt(diag(D'*D)));
% D =  bsxfun(@times,D, 1./sqrt(sum(D.^2))); %Normalization due to numerical things

