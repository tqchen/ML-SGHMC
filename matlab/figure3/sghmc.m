function [ data ] = sghmc( gradU, eta, L, alpha, x, V )
%% SGHMC using gradU, for L steps, starting at position x,  
%% return data: array of positions
m = length(x);
data = zeros( m, L );
beta = V * eta * 0.5;

if beta > alpha
    error('too big eta');
end

sigma = sqrt( 2 * eta * (alpha-beta) );
p = randn( m, 1 ) * sqrt( eta ); 
momentum = 1 - alpha;

for t = 1 : L
    p = p * momentum - gradU( x ) * eta + randn(2,1)* sigma;
    x = x + p;
    data(:,t) = x;
end

end
