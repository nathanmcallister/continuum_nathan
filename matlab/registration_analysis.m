%% Rigid Registration Analysis
% Cameron Wolfe 2/26/2024
%
% Performs a rigid registration between the model and aurora frame,
% providing T_aurora_2_model.  Then performas a rigid registration to
% find T_coil_2_tip, which can then be used to find T_tip_2_model
% (the tip transform) for this static case.  T_tip_2_coil

%% Setup
% Input filenames
SW_MODEL_POS_FILE = "../tools/5_model_registration_points_in_sw";
SW_TIP_POS_FILE = "../tools/all_tip_registration_points_in_sw";

PEN_FILE = "../tools/penprobe";
REG_FILE = "../data/reg_hat_all.csv";

T_SW_2_MODEL_FILE = "../tools/T_sw_2_model";
T_SW_2_TIP_FILE = "../tools/T_sw_2_tip";

% Output filenames
T_AURORA_2_MODEL_FILE = "../tools/T_aurora_2_model";
T_TIP_2_COIL_FILE = "../tools/T_tip_2_coil";

%% File inputs
model_reg_pos_in_sw = readmatrix(SW_MODEL_POS_FILE);
tip_reg_pos_in_sw = readmatrix(SW_TIP_POS_FILE);

T_sw_2_model = readmatrix(T_SW_2_MODEL_FILE);
T_sw_2_tip = readmatrix(T_SW_2_TIP_FILE);

model_reg_pos_in_model = T_mult(T_sw_2_model, model_reg_pos_in_sw);
tip_reg_pos_in_model = T_mult(T_sw_2_model, tip_reg_pos_in_sw);
tip_reg_pos_in_tip = T_mult(T_sw_2_tip, tip_reg_pos_in_sw);

pen_tip_pos = readmatrix(PEN_FILE);

reg_table = readtable(REG_FILE);
pen_table = reg_table(1:2:end, :);
tip_table = reg_table(2:2:end, :);

pen_transforms = table2array(pen_table(:, 4:10));
tip_transforms = table2array(tip_table(:, 4:10));

pen_quat = pen_transforms(:, 1:4)';
pen_pos = pen_transforms(:, 5:end)';

tip_quat = tip_transforms(:, 1:4)';
tip_pos = tip_transforms(:, 5:end)';

reg_measurements_in_aurora = nan(3, size(pen_transforms, 1));
num_reg_measurements = size(reg_measurements_in_aurora, 2);

num_model_reg_measurements = length(model_reg_pos_in_sw);
num_tip_reg_measurements = length(tip_reg_pos_in_sw);

for i=1:num_reg_measurements
    reg_measurements_in_aurora(:, i) = pen_pos(:, i) + quatrotate(pen_quat(:, i)', pen_tip_pos)';
end

model_reg_measurements_in_aurora = reg_measurements_in_aurora(:, 1:num_model_reg_measurements);
tip_reg_measurements_in_aurora = reg_measurements_in_aurora(:, end-num_tip_reg_measurements+1:end);

[~, T_aurora_2_model_model, rmse_model] = rigid_align_svd(model_reg_measurements_in_aurora, model_reg_pos_in_model)
[~, T_aurora_2_model_tip, rmse_tip] = rigid_align_svd(tip_reg_measurements_in_aurora, tip_reg_pos_in_model)
[~, T_aurora_2_model_all, rmse_all] = rigid_align_svd([model_reg_measurements_in_aurora, tip_reg_measurements_in_aurora], [model_reg_pos_in_model, tip_reg_pos_in_model])

model_reg_measurements_in_model_model = T_mult(T_aurora_2_model_model, model_reg_measurements_in_aurora);
tip_reg_measurements_in_model_model = T_mult(T_aurora_2_model_model, tip_reg_measurements_in_aurora);
model_reg_measurements_in_model_tip = T_mult(T_aurora_2_model_tip, model_reg_measurements_in_aurora);
tip_reg_measurements_in_model_tip = T_mult(T_aurora_2_model_tip, tip_reg_measurements_in_aurora);
model_reg_measurements_in_model_all = T_mult(T_aurora_2_model_all, model_reg_measurements_in_aurora);
tip_reg_measurements_in_model_all = T_mult(T_aurora_2_model_all, tip_reg_measurements_in_aurora);

close all;
figure(1);
plot3(model_reg_measurements_in_aurora(1,:)', model_reg_measurements_in_aurora(2,:)', model_reg_measurements_in_aurora(3,:)', 'x');
hold on;
plot3(tip_reg_measurements_in_aurora(1,:)', tip_reg_measurements_in_aurora(2,:)', tip_reg_measurements_in_aurora(3,:)', 'x');
hold off;

figure(2);
plot3(model_reg_pos_in_model(1,:)', model_reg_pos_in_model(2,:)', model_reg_pos_in_model(3,:)', 'x');
hold on;
plot3(tip_reg_pos_in_model(1,:)', tip_reg_pos_in_model(2,:)', tip_reg_pos_in_model(3,:)', 'x');
plot3(model_reg_measurements_in_model_tip(1,:)', model_reg_measurements_in_model_tip(2,:)', model_reg_measurements_in_model_tip(3,:)', 'x');
plot3(tip_reg_measurements_in_model_tip(1,:)', tip_reg_measurements_in_model_tip(2,:)', tip_reg_measurements_in_model_tip(3,:)', 'x');
hold off
