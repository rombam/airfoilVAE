% Plot an airfoil from output_airfoil.dat
A = load("output_airfoil.dat");
B = load("unsmooth_airfoil.dat");
figure
plot(A(:,1), A(:,2), '-o', B(:,1), B(:,2), '-o')
%plot(A(:,1), A(:,2), '-o')
grid minor
legend('Smoothed','Original')
xlim([-0.1 1.1])