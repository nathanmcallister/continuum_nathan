%% Hat Rigid Registration V2
% Cameron Wolfe 03/26/2024
%
% Performs a rigid registration between the model and aurora frame,
% providing T_aurora_2_model.  Then performas a rigid registration to
% find T_coil_2_tip, which can then be used to find T_tip_2_model
% (the tip transform) for this static case.  T_tip_2_coil

%% Setup
% Input filenames
SW_MODEL_POS_FILE = "../tools/model_registration_points_in_sw";
SW_TIP_POS_FILE = "../tools/all_tip_registration_points_in_sw";

PEN_FILE = "../tools/penprobe7";
REG_FILE = "../data/reg_03_26_24c.csv";

T_SW_2_MODEL_FILE = "../tools/T_sw_2_model";
T_SW_2_TIP_FILE = "../tools/T_sw_2_tip";

% Output filenames
T_AURORA_2_MODEL_FILE = "../tools/T_aurora_2_model";
T_TIP_2_COIL_FILE = "../tools/T_tip_2_coil";

%% File inputs
model_reg_truth_in_sw = readmatrix(SW_MODEL_POS_FILE);
tip_reg_truth_in_sw = readmatrix(SW_TIP_POS_FILE);

T_sw_2_model = readmatrix(T_SW_2_MODEL_FILE);
T_sw_2_tip = readmatrix(T_SW_2_TIP_FILE);

penprobe = readmatrix(PEN_FILE);
reg_table = readtable(REG_FILE);

%% Get truth registration positions
model_reg_truth_in_model = T_mult(T_sw_2_model, model_reg_truth_in_sw);
tip_reg_truth_in_tip = T_mult(T_sw_2_tip, tip_reg_truth_in_sw);

%% Extracting and processing aurora data
pen_mat = table2array(reg_table(strcmp('0A', reg_table{:, 3}), 4:10));
tip_mat = table2array(reg_table(strcmp('0B', reg_table{:, 3}), 4:10));

% Use penprobe to get pen tip position
pen_positions = nan(3, size(pen_mat,1));
for i=1:size(pen_mat, 1)
    pen_positions(:, i) = (pen_mat(i, 5:7) + quatrotate(pen_mat(i, 1:4), penprobe))';
end

% Split into model registration points and tip registration points
model_reg_meas_in_aurora = pen_positions(:, 1:size(model_reg_truth_in_model, 2));
tip_reg_meas_in_aurora = pen_positions(:, size(model_reg_truth_in_model, 2)+1:end);

%% SVD rigid registrations
[~, T_aurora_2_model, rmse_aurora_2_model] = rigid_align_svd(model_reg_meas_in_aurora, model_reg_truth_in_model)

[~, T_aurora_2_tip, rmse_aurora_2_tip] = rigid_align_svd(tip_reg_meas_in_aurora, tip_reg_truth_in_tip)

%% Coil to Aurora registration
mean_tip_quat = quat_mean(tip_mat(:,1:4));
mean_tip_pos = mean(tip_mat(:, 5:end));

T_coil_2_aurora = [[quat2dcm(mean_tip_quat), mean_tip_pos']; [0 0 0 1]]
rmse_coil_2_aurora = sqrt(mean((tip_mat(:,5:end) - mean_tip_pos).^2, 'all'))

%% Final transforms
T_tip_2_coil = T_coil_2_aurora^-1 * T_aurora_2_tip^-1
T_tip_2_model = T_aurora_2_model * T_coil_2_aurora * T_tip_2_coil

T_tip_2_model_truth = T_sw_2_model * T_sw_2_tip^-1

%% File outputs
writematrix(T_aurora_2_model, T_AURORA_2_MODEL_FILE);
system(("mv " + T_AURORA_2_MODEL_FILE + ".txt " + T_AURORA_2_MODEL_FILE)); % Get rid of .txt
writematrix(T_tip_2_coil, T_TIP_2_COIL_FILE);
system(("mv " + T_TIP_2_COIL_FILE + ".txt " + T_TIP_2_COIL_FILE)); % Get rid of .txt
