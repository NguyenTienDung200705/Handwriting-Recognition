%% Euler method and Heun method visualization using Matlab
% Differential equation: y' = -2x^3 + 12x^2 - 20x + 8.5
% Initial condition:	 y(0) = y0 = 1
% Examining range:		 [0, 4]

syms x y(x);
pt = diff(y) == -2*x^3 + 12*x^2 - 20*x + 8.5; % Differential equation
a = 0; b = 4;		% Range of example
y0 = 1;
dk = y(a) == y0;	% Initial condition
yt = dsolve(pt, dk, 'x'); % Using dsolve to find symbolic solution

figure(3);
h = fplot(yt, [a b], '-b'); hold on;
set(h, 'linewidth', 3);

f = @(x) -2*x^3 + 12*x^2 - 20*x + 8.5;
h = 0.2; xi = a : h : b;

yE = y0;
yH = y0;
for k = 1 : length(xi) - 1
    % Using Euler method to find numerical solution
    yE(k+1) = yE(k) + h * f(xi(k));
    
    dk = y(xi(k+1)) == yE(k+1);
    yt = dsolve(pt, dk, 'x'); % For each Euler point, find another 'accurate' solution
    fplot(yt, [a b], '--b');
    plot(xi(k+1), yE(k+1), '*g');
    
    % Using Heun method to find numerical solution
    yH(k+1) = yH(k) + h * (f(xi(k)) + f(xi(k+1))) / 2;
    
    dk = y(xi(k+1)) == yH(k+1);
    yt = dsolve(pt, dk, 'x'); % For each Heun  point, find another 'accurate' solution
    fplot(yt,[a b],'--b');
    plot(xi(k+1), yH(k+1), '*m');
end

% Highlighting numerical solutions
h = plot(xi, yE, 'g'); set(h, 'linewidth', 3);
h = plot(xi, yH, 'm'); set(h, 'linewidth', 3);
ylim([1 9]); hold off;
