import json


def evaluate_bc5(eval_data, get_map=False):
    eval_map = json.load(open('./data/cdr_test_evaluate.txt', 'r'))

    fn = sum([len(eval_map[k]) for k in eval_map])
    fn_m = {k: list(eval_map[k]) for k in eval_map}
    fp = 0
    fp_m = {k: [] for k in eval_map}
    tp = 0

    for k in eval_data:
        pm_data = eval_data[k]
        for i in pm_data:
            if i in eval_map[k]:
                tp += 1
                fn -= 1
                fn_m[k].remove(i)
            else:
                fp += 1
                fp_m[k].append(i)

    if tp == 0:
        return 0, 0, 0, tp, fp, fn

    p = float(tp) / (tp + fp)
    r = float(tp) / (tp + fn)
    f1 = 2 * p * r / (p + r)

    if get_map:
        return p, r, f1, tp, fp, fn, fn_m, fp_m
    else:
        return p, r, f1, tp, fp, fn