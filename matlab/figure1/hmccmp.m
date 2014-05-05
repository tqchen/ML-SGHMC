%% compare different HMC approaches
figure();
%% draw probability diagram
xGrid = [-3:xStep:3];
y = exp( - U(xGrid) );
y = y / sum(y) / xStep;
plot(xGrid,y);
hold on;

%% HMC without noise with M-H
samples = zeros(nsample,1);
x = 0;
for i = 1:nsample
    x = hmc( U, gradUPerfect, m, dt, nstep, x, 1);
    samples(i) = x;
end
[yhmc,xhmc] = hist(samples, xGrid);
yhmc = yhmc / sum(yhmc) / xStep;
plot( xhmc, yhmc, 'c-v');

%% HMC without noise: no M-H
samples = zeros(nsample,1);
x = 0;
for i = 1:nsample
    x = hmc( U, gradUPerfect, m, dt, nstep, x, 0);
    samples(i) = x;
end
[yhmc,xhmc] = hist(samples, xGrid);
yhmc = yhmc / sum(yhmc) / xStep;
plot( xhmc, yhmc, 'r-.');

%% HMC with noise, with M-H
samples = zeros(nsample,1);
x = 0;
for i = 1:nsample
    x = hmc( U, gradU, m, dt, nstep, x, 1);
    samples(i) = x;
end
[yhmc,xhmc] = hist(samples, xGrid);
yhmc = yhmc / sum(yhmc) / xStep;
plot( xhmc, yhmc, 'm-x');

%% HMC with noise, no M-H
samples = zeros(nsample,1);
x = 0;
for i = 1:nsample
    x = hmc( U, gradU, m, dt, nstep, x, 0);
    samples(i) = x;
end
[yhmc,xhmc] = hist(samples, xGrid);
yhmc = yhmc / sum(yhmc) / xStep;
plot( xhmc, yhmc, 'k-.');

%% SGHMC with noise, no M-H
samples = zeros(nsample,1);
x = 0;
for i = 1:nsample
    x = sghmc( U, gradU, m, dt, nstep, x, C, V );
    samples(i) = x;
end
[yhmc,xhmc] = hist(samples, xGrid);
yhmc = yhmc / sum(yhmc) / xStep;
plot( xhmc, yhmc, 'g');

%% Plot graph
legend( 'True Distribution','HMC without noise(with M-H)', 'HMC without noise(no M-H)', 'HMC with noise(with M-H)','HMC with noise(no M-H)', 'SGHMC' );

len = 5;
set(gcf, 'PaperPosition', [0 0 len len/8.0*6.5] )
set(gcf, 'PaperSize', [len len/8.0*6.5] )
xlabel('\theta');
saveas( gcf, fgname, 'fig');
