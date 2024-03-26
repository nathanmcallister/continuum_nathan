% Filenames
TIP_FILENAME = "../tools/penprobe";
GRID_REG_FILENAME = "../tools/grid_line_reg";
GRID_MEAS_FILENAME = "../data/grid_line.csv";

tip = readmatrix(TIP_FILENAME);

pos_reg = readmatrix(GRID_REG_FILENAME);

% File parsing
reg_table = readtable(GRID_MEAS_FILENAME,'Delimiter',',');
reg_table = reg_table(1:height(unique(reg_table(:,3))):end, :);

reg_matrix = table2array(reg_table(:, 4:10));
q = reg_matrix(:, 1:4);
t = reg_matrix(:, 5:end);
pos_meas = nan(size(pos_reg));
for i=1:size(pos_reg, 2)
    pos_meas(:,i) = (t(i,:) + quatrotate(q(i,: ), tip))';
end

[~, T, rmse] = rigid_align_svd(pos_meas, pos_reg)

x_dist_points = [7 18:25];
y_dist_points = 9:17;

x_dist_reg = sqrt(sum((pos_reg(:, x_dist_points) - pos_reg(:, 1)).^2, 1));
y_dist_reg = sqrt(sum((pos_reg(:, y_dist_points) - pos_reg(:, 1)).^2, 1));
x_dist_meas = sqrt(sum((pos_meas(:, x_dist_points) - pos_meas(:, 1)).^2, 1));
y_dist_meas = sqrt(sum((pos_meas(:, y_dist_points) - pos_meas(:, 1)).^2, 1));

x_dist_ratio = x_dist_meas ./ x_dist_reg;
y_dist_ratio = y_dist_meas ./ y_dist_reg;
