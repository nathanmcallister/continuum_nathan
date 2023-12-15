clear all;

SERIALPORT_ARDUINO = "COM3";
fser_arduino = serialport(SERIALPORT_ARDUINO,115200);
flush(fser_arduino);

l_0 = 56; % Initial length in mm
d = 4; % Distance from center of spine to tendons in mm
n = 10; % number of segments

disc_diameter = 15; % (mm) diameter of the disk
PWMrange = 500-100;
setmid = 350;

r_vals = 4:4:20;
phi_vals = 0:pi/5:2*pi;

desired = [];
dls = [];
pwms = [];

for r_index = 1:length(r_vals)
    for phi_index = 1:length(phi_vals)
        
        desired(end+1,:) = [0,0];

        dl = inverse_kinematics(0, 0, l_0, d);
        dls(end+1, :) = dl;

        pwm = dl_2_pwm(dl, disc_diameter, PWMrange, setmid);
        pwms(end+1,:) = pwm;

        write(fser_arduino, pwm, "uint16");
        pause(3);
                
        r = r_vals(r_index);
        phi = phi_vals(phi_index);
        x = r * cos(phi);
        y = r * sin(phi);   

        dl = inverse_kinematics(x, y, l_0, d);
        dls(end+1, :) = dl;

        pwm = dl_2_pwm(dl, disc_diameter, PWMrange, setmid);
        pwms(end+1,:) = pwm;

        write(fser_arduino, pwm, "uint16");
        pause(3);
    end
end

dl = [-0 -0 -0 -0];

dtheta = dl(1:4)./(disc_diameter/2);
dPWM = -dtheta*PWMrange/pi;

u = ones(1,4) * setmid + dPWM;

write(fser_arduino,u,"uint16")


[l_1, l_2, l_3, l_4] = delta_2_length(l_0, dl);
[l, phi, kappa] = robot_specific_transformation(d, l_1, l_2, l_3, l_4);
T = generate_transformation_matrix(l, phi, kappa)

sub_length = linspace(0, l, 100);
pos = zeros(3,length(sub_length));

for i=1:length(pos)
    T_prime = generate_transformation_matrix(sub_length(i), phi, kappa);
    pos(:,i) = T_prime(1:3,4);
end

plot3(pos(1,:), pos(2,:), pos(3,:))
axis equal

disp([dl, l, phi, kappa,norm(T(1:2,4) - [x;y])])

clear fser_arduino;

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