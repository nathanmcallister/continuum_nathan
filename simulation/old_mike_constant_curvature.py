def mike_constant_curvature(
    delta_ls: List[Tuple[float, ...]], d: float, l: float
) -> List[Tuple[float, ...]]:
    segment_params = []
    tendon_pairs = [(0, 1), (0, 3), (2, 1), (2, 3)]
    cumulative_length = [0] * 4
    for segment in delta_ls:
        segment = list(segment)

        for i in range(4):
            segment[i] -= cumulative_length[i]

        tendon_lengths = [0] * 4

        for tendon_pair in tendon_pairs:
            dl1 = (
                -segment[tendon_pair[0]]
                if tendon_pair[0] == 0
                else segment[tendon_pair[0]]
            )
            dl2 = (
                -segment[tendon_pair[1]]
                if tendon_pair[1] == 1
                else segment[tendon_pair[1]]
            )

            theta = 1 / d * sqrt(dl1**2 + dl2**2)
            phi = atan2(dl2, dl1)

            tendon_lengths = [
                -d * theta * cos(phi),
                -d * theta * sin(phi),
                d * theta * cos(phi),
                d * theta * sin(phi),
            ]

            is_valid_length = True

            for i in range(4):
                if tendon_lengths[i] > segment[i]:
                    is_valid_length = False

            if is_valid_length:
                break

        theta = 1 / d * sqrt(segment[0] ** 2 + segment[1] ** 2)
        phi = atan2(-segment[1], -segment[0])
        kappa = theta / l

        for i in range(4):
            cumulative_length[i] += tendon_lengths[i]

        segment_params.append((l, kappa, phi))

    return segment_params


def mike_constant_curvature_inverse(
    segment_params: List[Tuple[float, ...]], d: float, l: float
) -> List[Tuple[float, ...]]:
    cumulative_delta = [0, 0, 0, 0]
    segment_dls = []

    for segment in segment_params:
        l = segment[0]
        kappa = segment[1]
        phi = segment[2]

        cumulative_delta[0] -= d * l * kappa * cos(phi)
        cumulative_delta[1] -= d * l * kappa * sin(phi)
        cumulative_delta[2] += d * l * kappa * cos(phi)
        cumulative_delta[3] += d * l * kappa * sin(phi)

        segment_dls.append(tuple(cumulative_delta))

    return segment_dls
