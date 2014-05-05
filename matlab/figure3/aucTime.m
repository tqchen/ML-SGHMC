function [ tau ] = aucTime( data, dvar )
%% auto correlation time, the calculation method 
%  comes from Hoffman et.al. No-U-Turn sampeler

m = size( data, 1 );
L = size( data, 2 );
% do not compute correlation when maxlag is too small
% we know data is zero mean
for  i = 1 : m
    acorr = xcorr( data(i,:), L, 'biased' ) / dvar;
    res = 1;
    for j = 1 : L
        rpho = 0.5 * (acorr( L + j )+ acorr( L - j ));
        if rpho < 0.05
            break
        end
        res = res + 2 * (1 - j / L); 
    end
    tau(i) = res;
end
