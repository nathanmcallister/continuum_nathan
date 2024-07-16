% restart
close all; clear; clc;

% change this!
range_of_motion = linspace(-pi, pi, 20);

L = 16*4; % mm
phi = linspace(0, pi, 5); phi(end) = [];
kappa = range_of_motion/L; % curvature

T = zeros(4,4);

figure;
hold on; grid on;
axis equal;
view([0 0]);
% generate transforms for each point
k = 1;
for i = 1 % 1:length(phi)
    for j = 1:length(kappa)
        disp((i-1)*length(kappa)+j);
        T(:,:,(i-1)*length(kappa)+j) = generate_transformation_matrix(L, phi(i), kappa(j));
        % plotTriad(T(:,:,k));
        plotTriad(T(:,:,k),3);
        k = k + 1;
        drawnow;
    end
end

for phi_idx = 1
    for theta_max = range_of_motion  % (2*pi/38)+2*(2*pi/19) % converting curvature to angle
        theta = linspace(0,theta_max,100);
        r = L/theta_max;
        arc_shape = [r-r*cos(theta);zeros(size(theta));r*sin(theta)];
        R = [cos(phi(phi_idx)) -sin(phi(phi_idx)) 0; sin(phi(phi_idx)) cos(phi(phi_idx)) 0; 0 0 1];
        arc_shape_rot = R*arc_shape;
        plot3(arc_shape_rot(1,:),arc_shape_rot(2,:),arc_shape_rot(3,:),'-','LineWidth',3,'Color',[0.8 0 0]);
        drawnow;
    end
end

% second segment transforms
T2 = zeros(4,4);

for phi_idx = 1
    for theta_max1 = range_of_motion
        for theta_max2 = range_of_motion
            theta2 = linspace(theta_max1,theta_max1+theta_max2,100);
            r1 = L/theta_max1;
            r2 = L/theta_max2;
            arc_shape = [(r1-r1*cos(theta_max1))+r2*cos(theta_max1)-r2*cos(theta2); zeros(size(theta2)); r1*sin(theta_max1)-(r2*sin(theta_max1))+r2*sin(theta2)];
            R = [cos(phi(phi_idx)) -sin(phi(phi_idx)) 0; sin(phi(phi_idx)) cos(phi(phi_idx)) 0; 0 0 1];
            arc_shape_rot = R*arc_shape;
            plot3(arc_shape_rot(1,:),arc_shape_rot(2,:),arc_shape_rot(3,:),'-','LineWidth',3,'Color',[0 (theta_max1+range_of_motion(1))/(2*range_of_motion(1)) (theta_max1+range_of_motion(1))/(2*range_of_motion(1))]);
            drawnow;
        end
    end
end

d = 1;
for l1 = 1:size(T, 3)
    for l2 = 1:size(T, 3)
        T2(:,:,d) = T(:,:,l2) * T(:,:,l1);
        plotTriad(T2(:,:,d), 3);
        d = d + 1;
        drawnow;
    end
end

xlabel("x (millimeters)")
ylabel("y (millimeters)")
zlabel("z (millimeters)")
axis equal;

hold off
%%
function pwm_cmds = dl_2_pwm(dl, disc_diameter, pwm_range, zero_point)
dtheta = 2 * dl / (disc_diameter);
dpwm = dtheta * pwm_range / pi;
pwm_cmds = ones(size(dpwm)) * zero_point + dpwm;
end

function dl = inverse_kinematics(x,y,l_0,d)

options = optimoptions('fmincon','Display','notify');

dl = fmincon(@min_func, [-.001,-.01,-.001,-.001], [1 1 1 1], [0], [], [], [], [], @(dl) cc_constraint(dl, x, y, l_0, d), options);

end

function [l_1, l_2, l_3, l_4] = delta_2_length(l_0, dl_1, dl_2, dl_3, dl_4)
if nargin == 2
    dl_2 = dl_1(2);
    dl_3 = dl_1(3);
    dl_4 = dl_1(4);
    dl_1 = dl_1(1);
end
l_1 = l_0 + dl_1;
l_2 = l_0 + dl_2;
l_3 = l_0 + dl_3;
l_4 = l_0 + dl_4;
end

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

function [c,ceq] = cc_constraint(dl, x, y, l_0, d)
phi = atan2(y,x);
a = sqrt(x^2+y^2);
[l_1, l_2, l_3, l_4] = delta_2_length(l_0, dl);
[l_out, phi_out, kappa_out] = robot_specific_transformation(d, l_1, l_2, l_3, l_4);
c = [];
if kappa_out == 0
    ceq = [phi - phi_out; -a];
else
    ceq = [phi - phi_out; (1-cos(l_out*kappa_out))/kappa_out - a];
end
end

function y = min_func(dl)
y = sum(abs(dl));
end

function [T_ext_Z] = extend_transfromation_z(T, L_ext)
z = T(3, 1:3); % pick out z direction
o_original = T(1:3, 4);
o_new = o_original' + L_ext * (z .* [-1 -1 1]); % why do i need flip the signs of x and y here?
T_ext_Z = T;
T_ext_Z(1:3,4) = o_new';
end

function TF_out = compute_dh_transform(TF_in, dh_row)

% initialize output transformation to input transformation
TF_out = TF_in;

% extract values for this particular joint
a           = dh_row(1);
alpha       = dh_row(2);
D           = dh_row(3);
theta       = dh_row(4);

% assemble transformation matrix
RZ = [cos(theta) -sin(theta) 0 0; sin(theta) cos(theta) 0 0; 0 0 1 0; 0 0 0 1];
RX = [1 0 0 0; 0 cos(alpha) -sin(alpha) 0; 0 sin(alpha) cos(alpha) 0; 0 0 0 1];
TZ = [1 0 0 0; 0 1 0 0; 0 0 1 D; 0 0 0 1];
TX = [1 0 0 a; 0 1 0 0; 0 0 1 0; 0 0 0 1];
T_KHALIL = RX*TX*RZ*TZ;
% T_DH = RZ*TZ*TX*RX;  % this is the standard DH convention, but ISI uses modification from Khalil 1986 paper
TF_out = TF_out*T_KHALIL;

end