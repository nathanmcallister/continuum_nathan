Ac = [0 1; -1 0];
Bc = [0; 1];
C = [1, 0];
D = [0];
dt = .1;

[A, B] = c2d(Ac, Bc, dt);

x = [[1;0] zeros(2,10)];
y = zeros(1,11);
for k=1:10
    y(:, k) = C * x(:,k);
    x(:, k+1) = A * x(:,k);
end
y(:, end) = C * x(:, end);