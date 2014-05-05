%% This file produces Figure 3a, running trace of SGLD and SGHMC
% compare samples, this script will take a long time to run

clear all;
global covS;
global invS;
V = 1;
% covariance matrix
rho = 0.9;
covS = [ 1, rho; rho, 1 ];
invS = inv( covS );
% intial x
x = [0;0];

% this is highest value tried so far for SGLD that does not diverge
etaSGLD = 0.18;
etaSGHMC = 0.05;
alpha = 0.05;

% number of steps 
L = 10000000;
nset = 5;
probUMap = @(X,Y) exp( - 0.5 *( X .* X * invS(1,1) + 2 * X.*Y*invS(1,2) + Y.* Y *invS(2,2) )) / ( 2*pi*sqrt(abs(det (covS))));   
funcU = @(x) 0.5 * x'*invS*x;
gradUTrue = @(x) invS * x;
gradUNoise = @(x) invS * x  + randn(2,1);

% set random seed
randn( 'seed',20 );

% do multiple experiment, record each sample

for i = 1 : nset
    eta = etaSGLD * (0.8^(i-1));
    dsgld = sgld( gradUNoise, eta, L, x, V );
    covESGLD(:,:,i) = dsgld * dsgld' / L;
    meanESGLD(:,i) = mean( dsgld, 2 );
    SGLDeta(i) = eta;
    SGLDauc(i) = mean(aucTime( dsgld, 1 ));
end

for i = 1 : nset
    dscale = (0.6^(i-1));
    eta = etaSGHMC * dscale*dscale;
    dsghmc = sghmc( gradUNoise, eta, L, alpha*dscale, x, V );
    covESGHMC(:,:,i) = dsghmc * dsghmc' / L;
    meanESGHMC(:,i) = mean( dsghmc, 2 );
    SGHMCeta(i) = eta;
    SGHMCauc(i) = mean(aucTime( dsghmc, 1 ));
end
save cmpdata.mat;
drawcmp;

