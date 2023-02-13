from sklearn.preprocessing import minmax_scale as sk_minmax_scale


def minmax_scale(seq, eps=1e-5):
    return sk_minmax_scale(seq) + eps


def two_seq_scale(seq1, seq2):
    seq_min = min(min(seq1), min(seq2))
    seq_max = max(max(seq1), max(seq2))

    seq1 = (seq1 - seq_min) / (seq_max - seq_min)
    seq2 = (seq2 - seq_min) / (seq_max - seq_min)

    return seq1, seq2



