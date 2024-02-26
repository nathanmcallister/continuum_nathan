%% Hat Rigid Registration V1
% Cameron Wolfe 2/19/2024
%
% Performs a rigid registration between the model and aurora frame,
% providing T_aurora_2_model.  Then performas a rigid registration to
% find T_coil_2_tip, which can then be used to find T_tip_2_model
% (the tip transform) for this static case.  T_tip_2_coil

%% Setup
% Input filenames
SW_MODEL_POS_FILE = "../tools/model_registration_points_in_sw";
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

for i=1:num_reg_measurements
    reg_measurements_in_aurora(:, i) = pen_pos(:, i) + quatrotate(pen_quat(:, i)', pen_tip_pos)';
end

model_reg_measurements_in_aurora = reg_measurements_in_aurora(:, 1:num_model_reg_measurements);
tip_reg_measurements_in_aurora = reg_measurements_in_aurora(:, num_model_reg_measurements+1:end);

%% Model frame registration
[~, T_aurora_2_model, aurora_2_model_rmse] = rigid_align_svd(model_reg_measurements_in_aurora, model_reg_pos_in_model);

disp("Aurora to Model Transform RMSE:");
disp(aurora_2_model_rmse);

disp("Aurora to Model Transformation Matrix:");
disp(T_aurora_2_model);

writematrix(T_aurora_2_model, T_AURORA_2_MODEL_FILE);
system(("mv " + T_AURORA_2_MODEL_FILE + ".txt " + T_AURORA_2_MODEL_FILE)); % Get rid of .txt

%% Tip frame registration
tip_reg_measurements_in_aurora_wrt_coil = tip_reg_measurements_in_aurora - tip_pos(:, num_model_reg_measurements+1:end);

tip_reg_measurements_in_coil_wrt_coil = nan(size(tip_reg_measurements_in_aurora_wrt_coil));
num_tip_reg_measurements = size(tip_reg_measurements_in_coil_wrt_coil, 2);

for i=1:num_tip_reg_measurements
    tip_reg_measurements_in_coil_wrt_coil(:, i) = quatrotate(quatinv(tip_quat(:, num_model_reg_measurements+i)'), tip_reg_measurements_in_aurora_wrt_coil(:, i)')';
end

[~, T_coil_2_tip, coil_2_tip_rmse] = rigid_align_svd(tip_reg_measurements_in_coil_wrt_coil, tip_reg_pos_in_tip);
T_tip_2_coil = T_coil_2_tip^-1;


disp("Coil to Tip Transform RMSE:");
disp(coil_2_tip_rmse);

disp("Coil to Tip Transformation Matrix:");
disp(T_coil_2_tip);

disp("Tip to Coil Transformation Matrix:");
disp(T_tip_2_coil);

writematrix(T_tip_2_coil, T_TIP_2_COIL_FILE);
system(("mv " + T_TIP_2_COIL_FILE + ".txt " + T_TIP_2_COIL_FILE)); % Get rid of .txt

%% Coil to Aurora Transform
mean_tip_quat = normalize(mean(tip_quat(:, 7:end), 2));
mean_tip_pos = mean(tip_pos(:, 7:end), 2);
T_coil_2_aurora = [[quat2dcm(mean_tip_quat'), mean_tip_pos]; [0 0 0 1]];

disp("Coil to Aurora Transformation Matrix:");
disp(T_coil_2_aurora);

%% Other tip frame registration
[~, T_aurora_2_tip, rmse_aurora_2_tip] = rigid_align_svd(tip_reg_measurements_in_aurora, tip_reg_pos_in_tip);
T_coil_2_tip_2 = (T_coil_2_aurora)^-1 * (T_aurora_2_tip)^-1;
disp("Coil to Tip Transformation Matrix V2:");
disp(T_coil_2_tip_2);

%% Tip to Model Transform (only for this static case as T_coil_2_aurora changes)
T_tip_2_model = T_aurora_2_model * T_coil_2_aurora * T_coil_2_tip^-1;
disp("Tip to Model Transformation Matrix:")
disp(T_tip_2_model);

T_tip_2_model_v2 = T_aurora_2_model * T_coil_2_aurora * (T_coil_2_tip_2)^-1;
disp("Tip to Model Transformation Matrix V2:")
disp(T_tip_2_model_v2);
