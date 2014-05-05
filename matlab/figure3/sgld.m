function [ data ] = sgld( gradU, eta, L, x, V )
%% SGLD using gradU, for L steps, starting at position x, use SGFS way to take noise gradient level into account, 
%% return data: array of positions
m = length(x);
data = zeros( m, L );
beta = V * eta * 0.5;
if  beta > 1
    error('too big eta');
end

sigma = sqrt( 2 * eta * (1-beta) );

for t = 1 : L
    dx = - gradU( x ) * eta + randn(2,1) * sigma;
    x = x + dx;
    data(:,t) = x;
end

end
