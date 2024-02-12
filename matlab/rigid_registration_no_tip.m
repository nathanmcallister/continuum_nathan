%% Rigid Registration No Tip
% Cameron Wolfe 2/5/2024
%
% Performs a rigid registration between the robot and aurora frame, as well
% as estimating the position and orientation of the robot tip from the CAD
%% Setup
% Filenames
TRUTH_FILE = "../tools/reg_truth_positions";
PEN_FILE = "../tools/penprobe_no_metal";
REG_FILE = "../data/reg.csv";

TIP_FILE = "../tools/tip_no_hat";

TIP_POSITION_IN_MODEL = [0;
    0;
    64];

%% File inputs
truth_pos = readmatrix(TRUTH_FILE);
pen_tip_pos = readmatrix(PEN_FILE);

reg_table = readtable(REG_FILE);

%% Data processing/ extraction
pen_table = reg_table(1:2:end, :);
pen_transforms = table2array(pen_table(:,4:10));
pen_q = pen_transforms(:, 1:4)';
pen_r = pen_transforms(:, 5:end)';

tip_table = reg_table(2:2:end, :);
tip_transforms = table2array(tip_table(:, 4:10));
tip_q = tip_transforms(:, 1:4)';
tip_r = tip_transforms(:, 5:end)';

pen_pos_aurora = nan(size(truth_pos));

num_points = size(truth_pos, 2);

for i=1:num_points
    pen_pos_aurora(:,i) = pen_r(:,i) + quatrotate(pen_q(:,i)', pen_tip_pos)';
end

%% Registration
[~, T_aurora_2_model, robot_rmse] = rigid_align_svd(pen_pos_aurora, truth_pos);
disp("Registration RMS Error");
disp(robot_rmse);

%% Tip Position Estimation
tip_pos_in_tip = nan(size(tip_pos_aurora));

for i=1:num_points
    T_tip_2_aurora = eye(4);
    T_tip_2_aurora(1:3, 1:3) = quat2dcm(tip_q(:, i)');
    T_tip_2_aurora(1:3, 4) = tip_r(:,i);
    tip_pos_in_tip(:,i) = rigid_transformation(inv(T_tip_2_aurora) * inv(T_aurora_2_model), TIP_POSITION_IN_MODEL);
end

tip_pos_in_tip = mean(tip_pos_in_tip, 2);
disp("Tip Position in Coil Frame:");
disp(tip_pos_in_tip);

%% File outputs
writematrix(tip_pos_in_tip, TIP_FILE);

% Remove .txt
system("mv " + TIP_FILE + ".txt " + TIP_FILE);

%% Functions
function transformed_point = rigid_transformation(TF, point)
point = [point; ones(1, size(point, 2))];
point = TF * point;
transformed_point = point(1:3,:);
end
