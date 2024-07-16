% restart
close all; clear; clc;

psi = pi/3; % change this!
L = 16*4; % mm

% both segments maintain same curvature
% doing this in xz plane for now

radial_span = linspace(0, L*2, 100);
radial_line = [radial_span*cos(psi); zeros(size(radial_span)); radial_span*sin(psi)];

figure;
hold on; grid on;
axis equal;
view([0 -1 0]);

plot3(radial_line(1,:), radial_line(2,:), radial_line(3,:), "-k")

% var initialization
T1 = zeros(4,4);
T2 = zeros(4,4);
theta_max = 0;

% case where theta = pi/2 - psi

theta_max1 = pi/2 - psi;
theta_max2 = theta_max1;
theta1 = linspace(0, theta_max1, 100);
theta2 = linspace(theta_max1, theta_max1+theta_max2, 100);
r1 = L/theta_max1;
r2 = L/theta_max2;
    
    % transforms
    T1(:,:,1) = generate_transformation_matrix(L, 0, 1/r1);
    T2(:,:,1) = generate_transformation_matrix(L, 0, 1/r2);
    Ttot = T2(:,:,1) * T1(:,:,1);
    plotTriad(Ttot(:,:,1), 3)

    % arcs
    arc1 = [r1-r1*cos(theta1);zeros(size(theta1));r1*sin(theta1)];
    arc2 = [(r1-r1*cos(theta_max1))+r2*cos(theta_max1)-r2*cos(theta2); zeros(size(theta2)); r1*sin(theta_max1)-(r2*sin(theta_max1))+r2*sin(theta2)];
    plot3(arc1(1,:),arc1(2,:),arc1(3,:),'-','LineWidth',3,'Color',[0.8 0 0]);
    plot3(arc2(1,:),arc2(2,:),arc2(3,:),'-','LineWidth',3,'Color',[0 0 0.8]);

drawnow;

% case where theta = pi - 2*psi
theta_max1 = pi - 2*psi;
theta_max2 = -theta_max1;
theta1 = linspace(0, theta_max1, 100);
theta2 = linspace(theta_max1, theta_max1+theta_max2, 100);
r1 = L/theta_max1;
r2 = L/theta_max2;
    
    % transforms
    T1(:,:,1) = generate_transformation_matrix(L, 0, -1/r1);
    T2(:,:,1) = generate_transformation_matrix(L, 0, -1/r2);
    Ttot = T2(:,:,1) * T1(:,:,1);
    plotTriad(Ttot(:,:,1), 3)

    % arcs
    arc1 = [r1-r1*cos(theta1);zeros(size(theta1));r1*sin(theta1)];
    arc2 = [(r1-r1*cos(theta_max1))+r2*cos(theta_max1)-r2*cos(theta2); zeros(size(theta2)); r1*sin(theta_max1)-(r2*sin(theta_max1))+r2*sin(theta2)];
    plot3(arc1(1,:),arc1(2,:),arc1(3,:),'-','LineWidth',3,'Color',[0.8 0 0]);
    plot3(arc2(1,:),arc2(2,:),arc2(3,:),'-','LineWidth',3,'Color',[0 0 0.8]);

xlabel("x (millimeters)")
ylabel("y (millimeters)")
zlabel("z (millimeters)")
axis equal;

hold off

%% functions
function [l, phi, kappa] = robot_specific_transformation(d, l_1, l_2, l_3, l_4)
l = (l_1 + l_2 + l_3 + l_4)/4;

kappa_x = (l_3-l_1)/(d*(l_1+l_3));
kappa_y = (l_4-l_2)/(d*(l_2+l_4));

kappa = sqrt(kappa_x^2 + kappa_y^2);

phi = atan2(kappa_y, kappa_x);
end

function T = generate_transformation_matrix(l, phi, kappa)
dh_matrix = [  phi    ,                      0, 0, -pi/2;
    kappa*l/2,                      0, 0,  pi/2;
    0, 2/kappa*sin(kappa*l/2), 0, -pi/2;
    kappa*l/2,                      0, 0,  pi/2;
    -phi,                      0, 0,     0];
if kappa == 0
    dh_matrix(3,2) = l;
end

T = eye(4);

for i=1:size(dh_matrix,1)
    %disp(T)
    T = T*generate_dh(dh_matrix(i,:));
end
%disp(T)
end

function T = generate_dh(dh_params)
theta = dh_params(1);
d     = dh_params(2);
r     = dh_params(3);
alpha = dh_params(4);

Z = [cos(theta), -sin(theta), 0, 0;
    sin(theta),  cos(theta), 0, 0;
    0,           0, 1, d;
    0,           0, 0, 1];

X = [1,          0,           0, r;
    0, cos(alpha), -sin(alpha), 0;
    0, sin(alpha),  cos(alpha), 0;
    0,          0,           0, 1];

T = Z*X;
end