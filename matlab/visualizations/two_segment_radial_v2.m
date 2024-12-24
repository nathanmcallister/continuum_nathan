% plot range of arm configurations that reach a radial line at angle psi
close all; clear; clc;close all; clear; clc;

psi = pi/3;
L = 16 * 4; % mm

figure;
hold on; grid on;
axis equal;
view([0 -1 0]);
plot_arcs(L, pi/3, -pi/3)
plot_arcs(L, pi/6, pi/6)
drawnow

% need a function to determine theta2 given psi, theta1, and L
function theta2 = find_theta2(L, psi, theta1)
    % playing around
    
end

function bool = is_reachable(L, psi, theta1)
% Determines whether the radial line is reachable given first segment
% INPUTS: 
%   L = segment lenght
%   theta1 = first segment angle
%   psi = radial angle
% OUTPUTS:
%   n = 1 if true, n = 0 if false

    r1 = L/theta1;
    t = r1 * sin(psi) / sin(pi-theta1-psi);
    n = r1 - t;
    if n <= 2*L/pi
        bool = 1;
    else
        bool = 0;
    end
end

function plot_arcs(L, theta_max1, theta_max2)
    theta1 = linspace(0, theta_max1, 100);
    theta2 = linspace(theta_max1, theta_max1+theta_max2, 100);
    r1 = L/theta_max1;
    r2 = L/theta_max2;
    arc1 = [r1-r1*cos(theta1);zeros(size(theta1));r1*sin(theta1)];
    arc2 = [(r1-r1*cos(theta_max1))+r2*cos(theta_max1)-r2*cos(theta2); zeros(size(theta2)); r1*sin(theta_max1)-(r2*sin(theta_max1))+r2*sin(theta2)];
    plot3(arc1(1,:),arc1(2,:),arc1(3,:),'-','LineWidth',3,'Color',[0.8 0 0]);
    plot3(arc2(1,:),arc2(2,:),arc2(3,:),'-','LineWidth',3,'Color',[0 0 0.8]);
    drawnow
end

