function dthetas = dls_2_dthetas(dls, wheel_radii, rotation_sign)
    if nargin == 2
        rotation_sign = 1;
    end

    dthetas = zeros(size(dls));
    assert(numel(wheel_radii) == size(dls, 1), "Number of wheels should match number of segments");
    
    assert(rotation_sign == 1 || rotation_sign == -1, "Sign should be positive or negative");

    dthetas = 


