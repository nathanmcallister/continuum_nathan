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

AURORA_2_ROBOT_FILENAME = "../tools/robot_reg";
TIP_FILENAME = "../tools/tip_reg_cad";

CAD_TIP_POSITION = [0;
                    0;
                    64];

%% File inputs
truth_pos = readmatrix(TRUTH_FILE);
pen_tip_pos = readmatrix(PEN_FILE);

reg_table = readtable(REG_FILE);

pen_table = reg_table(1:2:end, :);
pen_transforms = table2array(pen_table(:,4:10));
pen_q = pen_transforms(:, 1:4)';
pen_r = pen_transforms(:, 5:end)';

tip_table = reg_table(2:2:end, :);
tip_transforms = table2array(tip_table(:, 4:10));
tip_q = tip_transforms(:, 1:4)';
tip_r = tip_transforms(:, 5:end)';

pen_pos_aurora = nan(size(truth_pos));
tip_pos_aurora = nan(size(truth_pos));

num_points = size(truth_pos, 2);

for i=1:num_points
    tip_pos_aurora(:,i) = tip_r(:,i);
    pen_pos_aurora(:,i) = pen_r(:,i) + quatrotate(pen_q(:,i)', pen_tip_pos)';
end

%% Registration
[~, robot_transform, robot_rmse] = rigid_align_svd(pen_pos_aurora, truth_pos);
disp("Registration RMS Error");
disp(robot_rmse);

aurora_2_robot_q = dcm2quat(robot_transform(1:3,1:3));
aurora_2_robot_r = robot_transform(1:3,4);

%% Tip Position Estimation

tip_pos_robot = nan(size(tip_pos_aurora));

for i=1:num_points
    tip_pos_robot(:,i) = quatrotate(aurora_2_robot_q, (tip_pos_aurora(:,i) + aurora_2_robot_r)')';
end


%% Outputs
writematrix([aurora_2_robot_q, aurora_2_robot_r', robot_rmse], AURORA_2_ROBOT_FILENAME);
