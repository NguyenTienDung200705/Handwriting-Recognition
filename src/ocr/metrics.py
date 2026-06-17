import editdistance


def compute_cer(preds, gts):

    total_dist = 0
    total_chars = 0

    for pred, gt in zip(preds, gts):

        total_dist += editdistance.eval(pred, gt)
        total_chars += len(gt)

    return total_dist / total_chars if total_chars > 0 else 0


def compute_wer(preds, gts):

    total_dist = 0
    total_words = 0

    for pred, gt in zip(preds, gts):

        total_dist += editdistance.eval(pred.split(), gt.split())
        total_words += len(gt.split())

    return total_dist / total_words if total_words > 0 else 0