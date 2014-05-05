function [ newx ] = sghmc( U, gradU, m, dt, nstep, x, C, V )
%% SGHMC using gradU, for nstep, starting at position x

p = randn( size(x) ) * sqrt( m );
B = 0.5 * V * dt; 
D = sqrt( 2 * (C-B) * dt );

for i = 1 : nstep
    p = p - gradU( x ) * dt  - p * C * dt  + randn(1)*D;
    x = x + p./m * dt;
end
newx = x;
end
