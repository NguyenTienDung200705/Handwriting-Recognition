import math
from collections import defaultdict

NINF = -float("inf")

def logsumexp(a):
    if not a:
        return NINF
    m = max(a)
    if m == NINF:
        return NINF
    return m + math.log(sum(math.exp(x - m) for x in a))


def prefix_beam_search(emission_log_prob, beam_size=10, blank=0):

    T, C = emission_log_prob.shape

    beams = [(tuple(), (0.0, NINF))]

    for t in range(T):

        new_beams = defaultdict(lambda: (NINF, NINF))

        for prefix, (p_b, p_nb) in beams:

            for c in range(C):

                log_p = emission_log_prob[t, c]

                end_t = prefix[-1] if prefix else None

                nb_p_b, nb_p_nb = new_beams[prefix]

                if c == blank:

                    new_beams[prefix] = (
                        logsumexp([nb_p_b, p_b + log_p, p_nb + log_p]),
                        nb_p_nb
                    )

                else:

                    if c == end_t:

                        new_beams[prefix] = (
                            nb_p_b,
                            logsumexp([nb_p_nb, p_b + log_p])
                        )

                    else:

                        new_prefix = prefix + (c,)

                        nb_p_b2, nb_p_nb2 = new_beams[new_prefix]

                        new_beams[new_prefix] = (
                            nb_p_b2,
                            logsumexp([nb_p_nb2, p_b + log_p, p_nb + log_p])
                        )

        beams = sorted(
            new_beams.items(),
            key=lambda x: logsumexp(x[1]),
            reverse=True
        )[:beam_size]

    return list(beams[0][0])

class BeamSearchDecoder:
    def __init__(self, beam_size=10, blank=0):
        self.beam_size = beam_size
        self.blank = blank

    def decode(self, emission_log_prob):

        best = prefix_beam_search(
            emission_log_prob,
            beam_size=self.beam_size,
            blank=self.blank
        )

        return best