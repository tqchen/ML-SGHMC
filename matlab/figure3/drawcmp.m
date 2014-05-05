clear all;
load cmpdata;

for i = 1 : nset
  SGLDCovErr(i)   = sum(sum(abs( covESGLD(:,:,i) - meanESGLD(:,i)*meanESGLD(:,i)' - covS ))) / 4;
  SGHMCCovErr(i)  = sum(sum(abs( covESGHMC(:,:,i) - meanESGHMC(:,i)*meanESGHMC(:,i)'- covS ))) / 4;
end

len = 4;
figure();
plot( SGLDauc, SGLDCovErr , 'b-x' );
hold on;
plot( SGHMCauc,  SGHMCCovErr, 'r-o' );
xlabel('Autocorrelation Time');
ylabel('Average Absolute Error of Sample Covariance');

legend( 'SGLD', 'SGHMC' );

fgname = 'figure/sgldcmp-cov';

set(gcf, 'PaperPosition', [0 0 len len/8.0*6.5] )
set(gcf, 'PaperSize', [len len/8.0*6.5] )

saveas( gcf, fgname, 'fig');
saveas( gcf, fgname, 'pdf');
