%% Rigid Registration
% Cameron Wolfe 03/26/2024
%
% Performs a rigid registration between the model and aurora frame,
% providing T_aurora_2_model.  Then performas a rigid registration to
% find T_coil_2_tip.  RMSE is returned for all transforms gotten from
% data.  Additionally, T_tip_2_model is calculated and displayed, but
% not returned.
%
% INPUTS:
%  - REG_FILE: (string) Aurora data file (0A: pen, 0B: coil)
%  - PEN_FILE: (string) Pen pivot_cal file
%  - output_files: (boolean) If output files are desired
%
% OUTPUTS:
%  - T_aurora_2_model: (4x4 double) Aurora to model frame transformation matrix
%  - T_tip_2_coil: (4x4 double) tip to coil frame transformation matrix
%  - rmse: (struct) Contains RMS error for various transformations

function [T_aurora_2_model, T_tip_2_coil, T_tip_2_model, rmse] = rigid_registration(REG_FILE, PEN_FILE, output_files, repetitions)

%% Input parsing
if nargin == 2
    output_files = false;
    repetitions = 1;
end
if nargin == 3
    repetitions = 1;
end

%% Setup
% Truth filenames
SW_MODEL_POS_FILE = "../tools/model_registration_points_in_sw";
SW_TIP_POS_FILE = "../tools/all_tip_registration_points_in_sw";

T_SW_2_MODEL_FILE = "../tools/T_sw_2_model";
T_SW_2_TIP_FILE = "../tools/T_sw_2_tip";

% RMSE error struct initialization
rmse = struct();

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

% If multiple repetitions of a point are collected, use all of them in reg
temp_model = zeros(3, length(model_reg_truth_in_model) * repetitions);
temp_tip = zeros(3, length(tip_reg_truth_in_tip) * repetitions);

for i=1:repetitions
    for j=1:length(model_reg_truth_in_model)
        temp_model(:, (j - 1) * repetitions + i) = model_reg_truth_in_model(:, j);
    end
    for j=1:length(tip_reg_truth_in_tip)
        temp_tip(:, (j - 1) * repetitions + i) = tip_reg_truth_in_tip(:, j);
    end
end

model_reg_truth_in_model = temp_model;
tip_reg_truth_in_tip = temp_tip;

%% Extracting and processing aurora data
pen_mat = table2array(reg_table(strcmp('0A', reg_table{:, 3}), 4:10));
tip_mat = table2array(reg_table(strcmp('0B', reg_table{:, 3}), 4:10));

% Use penprobe to get pen tip position
pen_positions = nan(3, size(pen_mat,1));
for i=1:size(pen_mat, 1)
    R = quat2matrix(pen_mat(i, 1:4));
    pen_positions(:, i) = pen_mat(i, 5:7)' + R * penprobe';
end

% Split into model registration points and tip registration points
model_reg_meas_in_aurora = pen_positions(:, 1:size(model_reg_truth_in_model, 2));
tip_reg_meas_in_aurora = pen_positions(:, size(model_reg_truth_in_model, 2)+1:end);

%% SVD rigid registrations
[~, T_aurora_2_model, rmse.aurora_2_model] = rigid_align_svd(model_reg_meas_in_aurora, model_reg_truth_in_model);

T_aurora_2_model

[~, T_aurora_2_tip, rmse.aurora_2_tip] = rigid_align_svd(tip_reg_meas_in_aurora, tip_reg_truth_in_tip);

T_aurora_2_tip

%% Coil to Aurora registration
mean_tip_quat = quat_mean(tip_mat(:,1:4));
mean_tip_pos = mean(tip_mat(:, 5:end));

T_coil_2_aurora = [[quat2matrix(mean_tip_quat), mean_tip_pos']; [0 0 0 1]]
rmse.coil_2_aurora = sqrt(mean((tip_mat(:,5:end) - mean_tip_pos).^2, 'all'));

%% Final transforms
T_tip_2_coil = T_coil_2_aurora^-1 * T_aurora_2_tip^-1
rmse.tip_2_coil_est = sqrt(rmse.coil_2_aurora^2 + rmse.aurora_2_tip^2);
T_tip_2_model = T_aurora_2_model * T_coil_2_aurora * T_tip_2_coil
rmse.tip_2_model_est = sqrt(rmse.aurora_2_model^2 + rmse.coil_2_aurora^2 + rmse.tip_2_coil_est^2);

T_tip_2_model_truth = T_sw_2_model * T_sw_2_tip^-1

%% File outputs
if output_files == true
    % Output filenames
    T_AURORA_2_MODEL_FILE = "../tools/T_aurora_2_model";
    T_TIP_2_COIL_FILE = "../tools/T_tip_2_coil";
    
    writematrix(T_aurora_2_model, T_AURORA_2_MODEL_FILE);
    system(("mv " + T_AURORA_2_MODEL_FILE + ".txt " + T_AURORA_2_MODEL_FILE)); % Get rid of .txt
    writematrix(T_tip_2_coil, T_TIP_2_COIL_FILE);
    system(("mv " + T_TIP_2_COIL_FILE + ".txt " + T_TIP_2_COIL_FILE)); % Get rid of .txt
end
end
