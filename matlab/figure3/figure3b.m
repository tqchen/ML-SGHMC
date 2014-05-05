%% This file produces Figure 3b, running trace of SGLD and SGHMC
%% parameters

clear all;
global covS;
global invS;

V = 1;
% covariance matrix
rho = 0.9;
covS = [ 1, rho; rho, 1 ];
invS = inv( covS );
% intial 
x = [0;0];

% this is highest value tried so far for SGLD that does not diverge
etaSGLD = 0.05;
etaSGHMC = 0.05;
alpha = 0.035;
% number of steps 
L = 50;

probUMap = @(X,Y) exp( - 0.5 *( X .* X * invS(1,1) + 2 * X.*Y*invS(1,2) + Y.* Y *invS(2,2) )) / ( 2*pi*sqrt(abs(det (covS))));   
funcU = @(x) 0.5 * x'*invS*x;
gradUTrue = @(x) invS * x;
gradUNoise = @(x) invS * x  + randn(2,1);

[XX,YY] = meshgrid( linspace(-2,2), linspace(-2,2) );
ZZ = probUMap( XX, YY );
contour( XX, YY, ZZ );
hold on;
% set random seed
randn( 'seed',20 );

dsgld = sgld( gradUNoise, etaSGLD, L, x, V );
dsghmc = sghmc( gradUNoise, etaSGHMC, L, alpha, x, V );
h1=scatter( dsgld(1,:), dsgld(2,:), 'bx');
h2=scatter( dsghmc(1,:), dsghmc(2,:), 'ro' );


xlabel('x');
ylabel('y');
legend([h1 h2], {'SGLD', 'SGHMC'});
axis([-2.1 3 -2.1 3]);
len = 4;
set(gcf, 'PaperPosition', [0 0 len len/8.0*6.5] )
set(gcf, 'PaperSize', [len len/8.0*6.5] )
fgname = 'figure/sgldcmp-run';
saveas( gcf, fgname, 'fig');
saveas( gcf, fgname, 'pdf');

