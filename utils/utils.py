import os
import sys
import logging
import csv
import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_logger(save_dir, name="run.log"):

    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(save_dir, name)
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0], correct.squeeze()


def compute_statistics(values):
    return {
        "mean": np.mean(values),
        "std": np.std(values, ddof=1)
    }



def csv_writter(path, data_name, model_name, metrics, results):
    with open(path, 'w', newline='',encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([data_name, model_name])

        for method, res in [("MSP_results", results)]:
            writer.writerow([method] + metrics)
            values = ["{:.2f}±{:.2f}".format(res[metric]["mean"], res[metric]["std"]) for metric in metrics]
            writer.writerow([''] + values)


def save_cifar10c_results_to_csv(save_path, metrics, cor_results_all_models):
    csv_file_path = os.path.join(save_path, 'cifar10c_results.csv')

    with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)

        # Write headers
        headers = ["Model", "Corruption", "Severity"] + metrics
        writer.writerow(headers)

        # Iterate over all models, corruptions, and severities
        for model_name, cor_results in cor_results_all_models.items():
            for corruption, severities in cor_results.items():
                for severity, results in severities.items():
                    values = [model_name, corruption, severity]
                    for metric in metrics:
                        values.append(f"{results[metric]:.2f}")
                    writer.writerow(values)



