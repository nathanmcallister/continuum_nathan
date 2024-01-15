# r \theta = l, (r - d) \theta = l - dl
# \theta = l / r, \theta = (l - dl) / (r - d)
# l / r = (l - dl) / (r - d)
# r / l = (r - d) / (l - dl)
# r / l - r / (l - dl) = - d / (l - dl)
# r (l - dl - l ) / (l (l - dl)) = - d / (l - dl)
# - r dl / (l (l - dl)) = - d / (l - dl)
# - r dl / l = - d
# r = d l / dl
# \kappa = dl / (d * l)

def one_tendon_constant_curvature(dl, segment_len, centroid_distance):
    return dl / (centroid_distance * segment_len)
