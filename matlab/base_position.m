T_AURORA_2_MODEL_FILE = "../tools/T_aurora_2_model";
PEN_FILE = "../tools/penprobe_03_28_24a";

ARC_FILE = "../data/base_test/base_arc.csv";
TOP_FILE = "../data/base_test/base_top.csv";


T_aurora_2_model = readmatrix(T_AURORA_2_MODEL_FILE);
pen = readmatrix(PEN_FILE);

arc_table = readtable(ARC_FILE);
top_table = readtable(TOP_FILE);

arc_mat = table2array(arc_table(:, 4:10));
top_mat = table2array(top_table(:, 4:10));

arc_quat = arc_mat(:, 1:4);
top_quat = top_mat(:, 1:4);

arc_pos = arc_mat(:, 5:end);
top_pos = top_mat(:, 5:end);

arc_pos_in_aurora = nan(3, height(arc_table));
top_pos_in_aurora = nan(3, height(top_table));

for i=1:height(arc_table)
    arc_pos_in_aurora(:, i) = (arc_pos(i,:) + quatrotate(arc_quat(i,:), pen))';
end

for i=1:height(top_table)
    top_pos_in_aurora(:, i) = (top_pos(i,:) + quatrotate(top_quat(i,:), pen))';
end

arc_pos_in_model = T_mult(T_aurora_2_model, arc_pos_in_aurora);
top_pos_in_model = T_mult(T_aurora_2_model, top_pos_in_aurora);

initial_center = [0 0]';

[arc_center, arc_rmse] = fminsearch(@(x) circle_error(x, arc_pos_in_model), initial_center);

initial_height = 0;

[top_height, top_rmse] = fminsearch(@(x) plane_error(x, top_pos_in_model), initial_height);

%% Plotting
theta = 0:pi/100:2*pi;
circle = 18 * [cos(theta); sin(theta)] + arc_center;
close all;
figure(1);
plot(arc_pos_in_model(1,:)', arc_pos_in_model(2,:)', 'x', 'DisplayName', 'Measured Edge of Base');
xlim([-25 25]);
ylim([-25 25]);
hold on;
plot(circle(1, :), circle(2, :), 'DisplayName', 'Fitted Edge of Base');
plot(arc_center(1), arc_center(2), '+', 'DisplayName', ['Center of Base (' num2str(arc_center(1)) ', ' num2str(arc_center(2)) ')'])
hold off;
legend('Location', 'northeast')
axis equal
title(['Measured and Fitted Location of Base (rms = ' num2str(arc_rmse) ' mm)'])
xlabel("x (mm)");
ylabel("y (mm)");

figure(2);
plot(top_pos_in_model(3,:)', 'x', 'DisplayName', 'Measured Height of Base')
xlim([0, size(top_pos_in_model, 2)]);
hold on;
plot([0 size(top_pos_in_model, 2)], mean(top_pos_in_model(3,:), 2) * [1 1], 'DisplayName', 'Fitted Height of Base');
hold off;
legend('Location', 'best');
title(['Measured and Fitted Height of Base (rms = ' num2str(top_rmse) ' mm)'])
xlabel("Point")
ylabel("z (mm)")

function rmse = circle_error(x, arc_pos)

if size(arc_pos, 1) == 3
    arc_pos = arc_pos(1:2, :);
end

shifted = arc_pos - x;
shifted_norm = sqrt(sum(shifted.^2, 1));
error = shifted_norm - 18;
rmse = sqrt(mean(error.^2));
end

function rmse = plane_error(x, top_pos)
rmse = sqrt(mean((top_pos(3,:) - x).^2, 2));
end
