%% Registration Test
lower_plane_file = "../data/lower_plane_2.csv";
upper_plane_file = "../data/upper_plane_2.csv";

penprobe_file = "../tools/penprobe";
penprobe = readmatrix(penprobe_file);

%% Input
lower_plane_table = readtable(lower_plane_file);
lower_plane_transforms = table2array(lower_plane_table(:, 4:10));
num_lower_plane_positions = length(lower_plane_transforms);
lower_plane_positions = nan(3, num_lower_plane_positions);

for i=1:num_lower_plane_positions
    lower_plane_positions(:, i) = (lower_plane_transforms(i, end-2:end) + quatrotate(lower_plane_transforms(i, 1:4), penprobe))';
end

upper_plane_table = readtable(upper_plane_file);
upper_plane_transforms = table2array(upper_plane_table(:, 4:10));
num_upper_plane_positions = length(upper_plane_transforms);
upper_plane_positions = nan(3, num_upper_plane_positions);

for i=1:num_upper_plane_positions
    upper_plane_positions(:, i) = (upper_plane_transforms(i, end-2:end) + quatrotate(upper_plane_transforms(i, 1:4), penprobe))';
end

%% Principal Axes Calculation
lower_mean = mean(lower_plane_positions, 2);
x_lower = lower_plane_positions - lower_mean;

x_lower_cov = x_lower * x_lower';

[lower_eigenvectors, lower_eigenvalues] = eig(x_lower_cov);
lower_eigenvalues = diag(lower_eigenvalues);

[~, smallest_lower_eigenvalue_idx] = min(lower_eigenvalues);

lower_plane_normal = lower_eigenvectors(:, smallest_lower_eigenvalue_idx);

upper_mean = mean(upper_plane_positions, 2);
x_upper = upper_plane_positions - upper_mean;

x_upper_cov = x_upper * x_upper';

[upper_eigenvectors, upper_eigenvalues] = eig(x_upper_cov);
upper_eigenvalues = diag(upper_eigenvalues);

[~, smallest_upper_eigenvalue_idx] = min(upper_eigenvalues);

upper_plane_normal = upper_eigenvectors(:, smallest_upper_eigenvalue_idx);

%% Distance between planes
upper_2_lower = lower_mean - upper_mean;
upper_projected_upper_2_lower = (upper_2_lower' * upper_plane_normal) * upper_plane_normal;
lower_projected_upper_2_lower = (upper_2_lower' * lower_plane_normal) * lower_plane_normal;
upper_projected_distance = norm(upper_projected_upper_2_lower);
lower_projected_distance = norm(lower_projected_upper_2_lower);
projected_distance = (upper_projected_distance + lower_projected_distance) / 2
%height = 72;
height = 76.75
delta = projected_distance - height

%% Normal vectors for plotting
vector_length = 10;
lower_plane_vector_end = lower_mean + vector_length * lower_plane_normal;
upper_plane_vector_end = upper_mean + vector_length * upper_plane_normal;

%% Plotting
co = colororder;
close all;
figure(1);
plot3(lower_plane_positions(1,:), lower_plane_positions(2,:), lower_plane_positions(3,:), 'x');
hold on;
plot3(upper_plane_positions(1,:), upper_plane_positions(2,:), upper_plane_positions(3,:), 'x');
quiver3(lower_mean(1), lower_mean(2), lower_mean(3), vector_length * lower_plane_normal(1), vector_length * lower_plane_normal(2), vector_length * lower_plane_normal(3), "Color", co(1,:), "LineWidth", 3);
quiver3(upper_mean(1), upper_mean(2), upper_mean(3), vector_length * upper_plane_normal(1), vector_length * upper_plane_normal(2), vector_length * upper_plane_normal(3), "Color", co(2,:), "LineWidth", 3);
hold off;


