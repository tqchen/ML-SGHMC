%% 
%% This file produces the Figure 1 in the paper in figure/func2 and figure/func4
% we only include figure/func4 due to space reason
% compare different HMC approaches under stochastic gradient and perfect
% gradient
%%
%% parameters
clear all;

nsample = 80000;
xStep = 0.1;
m = 1;
C = 3;
dt = 0.1;
nstep = 50;
V = 4;

% set random seed
randn('seed',10);

%% set up functions 
U = @(x) (-2* x.^2 + x.^4);
gradU = @(x) ( -4* x +  4*x.^3) +  randn(1) * 2;
gradUPerfect =  @(x) ( - 4*x +  4*x.^3 );
fgname = 'figure/func4';
hmccmp;

%% set up functions x^2;
U = @(x) 0.5 * x.^2;
gradU = @(x) x  +  randn(1) * 2;
gradUPerfect =  @(x) x ;
fgname = 'figure/func2';
hmccmp;


%% fine tune figures
h = hgload( 'figure/func2' );
axis([-3 3 0 0.65]);
saveas( gcf, 'figure/func2', 'pdf');

h = hgload( 'figure/func4' );
axis([-2 2 0 0.8]);
xland = get(gca, 'xlabel');
set( xland, 'fontsize', 20 );

saveas( gcf, 'figure/func4', 'pdf');
