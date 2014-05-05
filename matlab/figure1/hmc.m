function [ newx ] = hmc( U, gradU, m, dt, nstep, x, mhtest )
%% HMC using gradU, for nstep, starting at position x

p = randn( size(x) ) * sqrt( m );
oldX = x;
oldEnergy = p' * m * p / 2 + U(x); 
% do leapfrog
for i = 1 : nstep
    p = p - gradU( x ) * dt / 2;
    x = x + p./m * dt;
    p = p - gradU( x ) * dt / 2;
end

p = -p;

% M-H test
if mhtest ~= 0
    newEnergy  = p' * m * p / 2 + U(x);

    if exp(oldEnergy- newEnergy) < rand(1)
        % reject
        x = oldX;
    end
end
newx = x;
end
