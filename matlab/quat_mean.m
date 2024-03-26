function mean_quat = quat_mean(quaternion_array)
array_size = size(quaternion_array);
if (array_size(1) == 4)
    mean_quat = mean(quaternion_array, 2);
else
    mean_quat = mean(quaternion_array, 1);
end
mean_quat = mean_quat / norm(mean_quat);
end
