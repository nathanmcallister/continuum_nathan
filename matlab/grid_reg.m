% Filenames
TIP_FILENAME = "../tools/penprobe_no_metal";
GRID_FILENAME = "../data/grid_no_metal.csv";

% Grid Parameters
NUM_ROWS = 3;
NUM_COLS = 7;
SPACING = 25.4;

% Setup truth values
x = (0:(NUM_COLS-1)) * SPACING;
y = -(0:(NUM_ROWS-1)) * SPACING;

[x_mesh, y_mesh] = meshgrid(x,y);

x_truth = reshape(x_mesh', [1, NUM_ROWS * NUM_COLS]);
y_truth = reshape(y_mesh', [1, NUM_ROWS * NUM_COLS]);

pos_truth = [x_truth; y_truth; zeros(1,NUM_ROWS * NUM_COLS)];

% File parsing
reg_table = readtable(GRID_FILENAME,'Delimiter',',');
reg_matrix = table2array(reg_table(:, 4:10));
q = reg_matrix(:, 1:4);
t = reg_matrix(:, 5:end);
pos_measured = nan(size(pos_truth));
for i=1:size(pos_truth, 2)
    pos_measured(:,i) = (t(i,:)' + quatrotate(q(i,: ), tip));
end

[~, ~, rmse] = rigid_align_svd(pos_measured, pos_truth)