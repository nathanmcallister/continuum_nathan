hat_labels = ["A", "B"];
hat_dates = ["03_26_24", "03_28_24"];
hat_date_dict = dictionary(hat_labels, hat_dates);
pen_labels = 1:7;
pen_files = ["penprobe1", "penprobe2", "penprobe3", "penprobe4", "penprobe5", "penprobe6", "penprobe7"];
treatments = ["a", "b", "c"];

hat_idx = 1:length(hat_dates);
pen_idx = 1:length(pen_files);
treatment_idx = 1:length(treatments);
[pen_mesh, treatment_mesh, hat_mesh] = meshgrid(pen_idx, treatment_idx, hat_idx);

hat_column = reshape(hat_mesh, [], 1);
pen_column = reshape(pen_mesh, [], 1);
treatment_column = reshape(treatment_mesh, [], 1);

angle_aurora_2_model = zeros(size(hat_column));
x_aurora_2_model = zeros(size(hat_column));
y_aurora_2_model = zeros(size(hat_column));
z_aurora_2_model = zeros(size(hat_column));

angle_tip_2_coil = zeros(size(hat_column));
x_tip_2_coil = zeros(size(hat_column));
y_tip_2_coil = zeros(size(hat_column));
z_tip_2_coil = zeros(size(hat_column));

angle_tip_2_model = zeros(size(hat_column));
x_tip_2_model = zeros(size(hat_column));
y_tip_2_model = zeros(size(hat_column));
z_tip_2_model = zeros(size(hat_column));

for i=1:length(hat_column)
    h = hat_column(i);
    p = pen_column(i);
    t = treatment_column(i);
    reg_file = "../data/reg_" + hat_date_dict(hat_labels(h)) + treatments(t) + ".csv";
    pen_file = "../tools/" + pen_files(p);
    
    [T_aurora_2_model, T_tip_2_coil, T_tip_2_model, rmse] = rigid_registration(reg_file, pen_file, false);
    
    q_aurora_2_model = dcm2quat(T_aurora_2_model(1:3, 1:3))';
    r_aurora_2_model = T_aurora_2_model(1:3,4);
    q_tip_2_coil = dcm2quat(T_tip_2_coil(1:3, 1:3))';
    r_tip_2_coil = T_tip_2_coil(1:3,4);
    q_tip_2_model = dcm2quat(T_tip_2_model(1:3, 1:3))';
    r_tip_2_model = T_tip_2_model(1:3,4);
    
    angle_aurora_2_model(i) = 2 * acos(q_aurora_2_model(1));
    x_aurora_2_model(i) = r_aurora_2_model(1);
    y_aurora_2_model(i) = r_aurora_2_model(2);
    z_aurora_2_model(i) = r_aurora_2_model(3);
    
    angle_tip_2_coil(i) = 2 * acos(q_tip_2_coil(1));
    x_tip_2_coil(i) = r_tip_2_coil(1);
    y_tip_2_coil(i) = r_tip_2_coil(2);
    z_tip_2_coil(i) = r_tip_2_coil(3);
    
    angle_tip_2_model(i) = 2 * acos(q_tip_2_model(1));
    x_tip_2_model(i) = r_tip_2_model(1);
    y_tip_2_model(i) = r_tip_2_model(2);
    z_tip_2_model(i) = r_tip_2_model(3);
end

data_table = table(hat_labels(hat_column)', pen_labels(pen_column)', treatments(treatment_column)', angle_aurora_2_model, x_aurora_2_model, y_aurora_2_model, z_aurora_2_model, angle_tip_2_coil, x_tip_2_coil, y_tip_2_coil, z_tip_2_coil, angle_tip_2_model, x_tip_2_model, y_tip_2_model, z_tip_2_model, 'VariableNames', ["Hat", "Pen", "Treatment", "A2M_Angle", "A2M_x", "A2M_y", "A2M_z", "T2C_Angle", "T2C_x", "T2C_y", "T2C_z", "T2M_Angle", "T2M_x", "T2M_y", "T2M_z"]);

p_x = anovan(data_table.T2M_x, {data_table.Hat, data_table.Pen}, "model", "interaction");
p_y = anovan(data_table.T2M_y, {data_table.Hat, data_table.Pen}, "model", "interaction");
p_z = anovan(data_table.T2M_z, {data_table.Hat, data_table.Pen}, "model", "interaction");
