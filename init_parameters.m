% Xian Wei, Research Group for Geometric Optimization and Machine Learning
% Muenchen, 2014. Contact: xian.wei@tum.de

function param = init_parameters(param)
size_ydata = param.D_size;

param.GlobalMethod = 'Steepes_gradient';
%param.GlobalMethod = 'CGA';

% param.grassmann.geodesic = 'Absil'; % manopt, Absil;                    book  title: Optimization Algorithms on Matrix Manifolds
% param.grassmann.geodesic.method = 'Absil exponential mapping';
% param.grassmann.geodesic.method = 'Absil Retraction';
% param.grassmann.geodesic = 'Alan'; % Riemannian exponential mapping;    paper title: The geometry of algorithms with orthogonality constraints
% param.grassmann.geodesic = 'Hao';  % Iterative trace ratio method ;     paper title: A Geometric Revisit to the Trace Quotient Problem
% %param.grassmann.geodesic = 'Martin'; % Second order approximation;      paper title: An Intrinsic CG Algorithm for Computing Dominant Subspaces

%1 initialize dictionary
    param.channel_initial_dic = 'DCT';
    param.channel_initial_dic = 'Random';
    param.channel_initial_dic = 'From_data';
    
 param.inco = 'Dinctionary is incoherent';
 
    K = size_ydata(2); %number of atoms
    bb =sqrt(size_ydata(1)*size_ydata(2)); %dim of dictionary
    k_lamda =[-3,-2,-1,0,1,2,3]; 
    % initial dictionary
    %1 DCT dic
    if strcmpi(param.channel_initial_dic, 'DCT')
        Pn=ceil(sqrt(K));
        DCT=zeros(bb,Pn);
        for k=0:1:Pn-1,
            V=cos([0:1:bb-1]'*k*pi/Pn);
            if k>0, V=V-mean(V); end;
            DCT(:,k+1)=V/norm(V);
        end;
        DCT=kron(DCT,DCT);
    
        param.D = DCT(:,1:K );
        clear DCT V Pn;
        % param.batchSize = 1;   % length of each dataset
        param.lamda1       = 0.5+0.025*k_lamda(4);    
        %param.lamda1       = 0.15;
        %param.lamda1       =  0.001; 
        param.lamda2      = 0;  
        
    elseif strcmpi(param.channel_initial_dic, 'Random')
        M = K; % one of Second of original data 
        N = size_ydata(1)*size_ydata(2); %1024
        Phi = randn(N,M);
        Phi = Phi./repmat(sqrt(sum(Phi.^2,1)),[N,1]);	
        param.D = Phi;
    
        param.lamda1       = 0.25+0.025*k_lamda(4);               
        param.lamda2       = 0;                 
    
    %3 Get dic from data
    else   
       % load Initial_dic.mat;
 %       param.lamda1       = 0.07;            
        param.lamda1       = param.lamda1;    
        param.lamda2      = param.lamda2;  
 %       param.lamda2      = 0;
        %param.InitialDic = D0;
    end
%2 initialize projector P
% A = randn(D_row);
% P = .5*(A+A');  
%r = fix(D_row/2);
% Man.P = grassmannfactory(D_row, Reduced_dims);  
% Man.D = spherefactory(D_row, D_col);
% problem.M = productmanifold(Man);
% P = Man.D.rand();

% initial P
[Q, unused] = qr(randn( param.D_size(2), param.Reduced_dims ), 0); 
%X(:, :, i) = Q;
P = Q*Q';
%(1/dot(v,v))*v*v'; % generate projector
param.P = P;
param.initial_ortho_projection = Q;


%3 initialize other parameters
     param.N               = size(param.D,1);        
     param.max_iter        = 20; % Maximal main iterations.
     param.verbose         = 1; % display commments
     param.dataPeak        = 1;
     param.targetMSE       = 1e-6;
     
     param.lambda3         = 1e-2;
%     param.paramDicUpdating = struct(...
%             'eps',    param.targetMSE, ...
%             'L',      floor(0.5*N)  ); 
    param.paramLasso      = struct(...
            'mode',   2, ...            
            'lambda', param.lamda1, ...
            'lambda2', param.lamda2, ...
            'L',      floor(0.9*param.N*10)  );
%     param.lamda3 = 0.1;
%     param.Inco = 1; % penalty for incoherence between B and D
%     param.lamda4 = 1e-5;
end