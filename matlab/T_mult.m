function p_b = T_mult(T_a_2_b, p_a)

p_a = [p_a; ones(1, size(p_a, 2))];
p_b = T_a_2_b * p_a;
p_b = p_b(1:3, :);

end
