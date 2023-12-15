close all; clear all;

% Load in files
TIP_FILENAME = "../datafiles/penprobe.tip";
REG_FILENAME = "../datafiles/reg.csv";

% Solidworks frame to model frame rotation matrix
R_sw_2_model = rotz(-45)*rotx(90);

% Registration points on model in solidworks coordinates.
% Origin is at start of helix (top of cylindrical base)
% Cylindrical base is 6.35 mm, and pedestal is 5 mm (giving -11.35 mm)
p_1_sw = [ 50 -5   0]';
p_2_sw = [  0 -5 -50]';
p_3_sw = [-50 -5   0]';
p_4_sw = [ 10 -5  50]';

% Get ground truth points used for registration
registration_truth = R_sw_2_model * [p_1_sw, p_2_sw, p_3_sw, p_4_sw];

% load tip file
penprobe_tip = load(TIP_FILENAME);

% load registration points from aurora and apply tip compensation
reg_table = readtable(REG_FILENAME,'Delimiter',',');
reg_matrix = table2array(reg_table(:, 4:10));
pen_matrix = reg_matrix(1:2:end,:);
spine_matrix = reg_matrix(2:2:end,:);
registration_measured = nan(size(registration_truth));
for pt_idx = 1:size(pen_matrix,1)
    q = pen_matrix(pt_idx,1:4);
    t = pen_matrix(pt_idx,5:7);
    registration_measured(:,pt_idx) = t + quatrotate(q,penprobe_tip);
end

% register aurora to model
[~,TF_aurora_to_model,rmse] = rigid_align_svd(registration_measured,registration_truth)

%spine tip in model space
spine_tip_modelspace = [0,0,56]';

%coil to aurora transform, this is just using the first coil point, could
%filter to use more accurate
spinetip_in_coilspace = zeros(3,size(spine_matrix,1));
for i=1:size(spine_matrix,1)
    qcoil = spine_matrix(i,1:4);
    tcoil = spine_matrix(i,5:7);
    TF_coil_to_aurora = eye(4);
    TF_coil_to_aurora(1:3,1:3) = quat2matrix(qcoil);
    TF_coil_to_aurora(1:3,4) = tcoil;
    
    %model to coil and find the spine tip in coil space
    TF_model_to_coil = inv(TF_coil_to_aurora)*inv(TF_aurora_to_model);
    spinetip_in_coilspace(:,i) = hTF(spine_tip_modelspace,TF_model_to_coil,0);
end

mean_spinetip_in_coilspace = mean(spinetip_in_coilspace,2)

coil_at_registration = tcoil + quatrotate(qcoil,mean_spinetip_in_coilspace');

writematrix(TF_aurora_to_model,'../datafiles/TF_aurora_to_model.csv');
writematrix(mean_spinetip_in_coilspace,'../datafiles/spinetip_in_coilspace.csv');