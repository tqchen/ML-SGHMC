%% 
%% This file produces the Figure 2 in the paper in trace
% draw trace of different kinds of dynamics
%%
clear all;
%% parameters
m = 1;
C = 3;
dt = 0.1;
fgname = 'figure/trace'
nstep = 300;
niter  = 50;
% noise in the gradien t
sigma = 0.5;

gradUPerfect = @(x) x;
gradU = @(x) x + randn(1)* sigma;

figure();

xstart = 1;
pstart = 0;

% set random seed
randn('seed',10);

%% Hamiltonian dynamics with noised gradient 
x = xstart;
p = pstart;
xs = zeros( nstep, 1 );
ys = zeros( nstep, 1 );
% do leapfrog
for i = 1 : nstep
    for j = 1: niter
        p = p - gradU( x ) * dt / 2;
        x = x + p./m * dt;
        p = p - gradU( x ) * dt / 2;
    end
    xs(i) = x;
    ys(i) = p;
end
plot( xs, ys, 'rv', 'MarkerSize', 3 );
hold on;

%% Hamiltonian dynamics with noised gradient 
x = xstart;
p = pstart;
xs = zeros( nstep, 1 ); 
ys = zeros( nstep, 1 );
% do leapfrog
for i = 1 : nstep
    p = randn( size(x) ) * sqrt( m );
    for j = 1: niter
        p = p - gradU( x ) * dt / 2;
        x = x + p./m * dt;
        p = p - gradU( x ) * dt / 2;
    end
    xs(i) = x;
    ys(i) = p;
end
plot( xs, ys, 'ko', 'MarkerSize', 3 );
hold on;


% set random seed
randn('seed',10);

%% Second order Langevin dynamics with noised gradient
x = xstart;
p = pstart;
xs = zeros( nstep, 1 );
ys = zeros( nstep, 1 );
Bhat =  0.5 * sigma^2 * dt;
D = sqrt( 2 * (C-Bhat) * dt );

% do leapfrog
for i = 1 : nstep
    for j = 1: niter
        p = p - gradU( x ) * dt  - p * C * dt  + randn(1)*D;
        x = x + p./m * dt;
    end
    xs(i) = x;
    ys(i) = p;
end
plot( xs, ys, 'gs', 'MarkerSize', 3 );
hold on;


%% Hamiltonian dynamics with perfect gradient 
randn('seed',10);
x = xstart;
p = pstart;
xs = zeros( nstep, 1 );
ys = zeros( nstep, 1 );
% do leapfrog
for i = 1 : nstep
    for j = 1: niter
        p = p - gradUPerfect( x ) * dt / 2;
        x = x + p./m * dt;
        p = p - gradUPerfect( x ) * dt / 2;
    end
    xs(i) = x;
    ys(i) = p;
end
plot( xs, ys, 'bx', 'MarkerSize', 4 );
hold on;

%% save the  figures
xlabel('\theta');
ylabel('r');

legend( 'Noisy Hamiltonian dynamics', 'Noisy Hamiltonian dynamics(resample r each 50 steps)','Noisy Hamiltonian dynamics with friction', 'Hamiltonian dynamics');

len =5.5;
set(gcf, 'PaperPosition', [0 0 len len/8.0*6.5] )
set(gcf, 'PaperSize', [len len/8.0*6.5] )
xland = get(gca, 'xlabel');
yland = get(gca, 'ylabel');
set( xland, 'fontsize', 20 );
set( yland, 'fontsize', 20 );

saveas(gcf,fgname, 'pdf');
saveas(gcf,fgname, 'fig');

