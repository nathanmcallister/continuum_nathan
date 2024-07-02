%% Tip Cal Comparison
% Cameron Wolfe 3/26/24
%
% Using tip cal data files, generate a penprobe tip file for each run
% and compare RMSE/ output to ensure aurora probe is functioning
% properly.

DATA_1_FILENAME = "../data/tip_cal_1.csv";
DATA_2_FILENAME = "../data/tip_cal_2.csv";
DATA_3_FILENAME = "../data/tip_cal_3.csv";
DATA_4_FILENAME = "../data/tip_cal_4.csv";
DATA_5_FILENAME = "../data/tip_cal_5.csv";
DATA_6_FILENAME = "../data/tip_cal_6.csv";
DATA_7_FILENAME = "../data/tip_cal_7.csv";

[tip_1, rmse_1] = pivot_cal_lsq(DATA_1_FILENAME, 3, "../tools/penprobe1");
[tip_2, rmse_2] = pivot_cal_lsq(DATA_2_FILENAME, 3, "../tools/penprobe2");
[tip_3, rmse_3] = pivot_cal_lsq(DATA_3_FILENAME, 3, "../tools/penprobe3");
[tip_4, rmse_4] = pivot_cal_lsq(DATA_4_FILENAME, 3, "../tools/penprobe4");
[tip_5, rmse_5] = pivot_cal_lsq(DATA_5_FILENAME, 3, "../tools/penprobe5");
[tip_6, rmse_6] = pivot_cal_lsq(DATA_6_FILENAME, 3, "../tools/penprobe6");
[tip_7, rmse_7] = pivot_cal_lsq(DATA_7_FILENAME, 3, "../tools/penprobe7");

rmses = [rmse_1 rmse_2 rmse_3 rmse_4 rmse_5 rmse_6 rmse_7]'
tips = [tip_1 tip_2 tip_3 tip_4 tip_5 tip_6 tip_7]'

% bingo bongo
