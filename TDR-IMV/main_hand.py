import os
import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data import handwritten0, Scene, YaleB, PIE, Caltech101all, ALOI100, leaves100, sources3, MNIST10k, Cora, Reuters1200, Wikipedia

from model import TDR_IMV

from util import mv_dataset, read_mymat, build_ad_dataset, process_data, get_validation_set, mv_tabular_collate, \
    get_validation_set_Sn, partial_mv_dataset, partial_mv_tabular_collate, Uncertainty_filter, create_label_mask
import time
import warnings
from EarlyStopping_hand import EarlyStopping
from collections import Counter
from select_k_neighbors import get_samples
from loss_function import get_loss
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATASET_REGISTRY = {
    'handwritten0':  handwritten0,
    'Scene':         Scene,
    'YaleB':         YaleB,
    'PIE':           PIE,
    'Caltech101all': Caltech101all,
    'ALOI100':       ALOI100,
    'leaves100':     leaves100,
    'sources3':      sources3,
    'MNIST10k':      MNIST10k,
    'Cora':          Cora,
    'Reuters1200':   Reuters1200,
    'Wikipedia':     Wikipedia,
}

class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def run_experiment(dataset_name_key, n_labeled_per_class, args):
    """Run n_runs experiments for one (dataset, n_labeled) combination.
    Returns (mean_acc, std_acc, all_results).
    """
    dataset = DATASET_REGISTRY[dataset_name_key]()
    num_classes = dataset.num_classes
    num_views   = dataset.num_views
    dims        = dataset.dims
    mat_name    = dataset.data_name + ".mat"
    delta = 1
    gamma = 1
    beta  = 1.5

    # ── One-time data loading ──
    np.random.seed(args.label_seed)
    X, Y, Sn = read_mymat('./data/', mat_name, ['X', 'Y'], args.missing_rate)
    partition = build_ad_dataset(Y, p=0.8, seed=999)
    X = process_data(X, num_views)
    Y_orig_train = np.squeeze(Y)[partition['train']]

    all_results = []
    t_start = time.time()

    for run in range(args.n_runs):
        run_seed = args.label_seed + run
        print(f"\n{'='*50}")
        print(f"[{dataset_name_key}] n_labeled={n_labeled_per_class}  "
              f"Run {run + 1}/{args.n_runs}  (seed={run_seed})")
        print(f"{'='*50}")

        # Label mask on original training samples (before imputation)
        label_mask_orig = create_label_mask(Y_orig_train, n_labeled_per_class, seed=run_seed)
        print(f"Weak supervision: {label_mask_orig.sum()} labeled / {len(label_mask_orig)} "
              f"original train samples ({n_labeled_per_class} per class, seed={run_seed})")
        for c in np.unique(Y_orig_train):
            class_labeled = np.where(label_mask_orig & (Y_orig_train == c))[0].tolist()
            print(f"  类别 {c}: {class_labeled}")

        # Imputation + label mask propagation
        X_train, Y_train, X_test, Y_test, Sn_train, label_mask_train = get_samples(
            x=X, y=Y, sn=Sn,
            train_index=partition['train'],
            test_index=partition['test'],
            n_sample=args.n_sample,
            k=args.k,
            label_mask=label_mask_orig)

        train_loader = DataLoader(
            dataset=partial_mv_dataset(X_train, Sn_train, Y_train, label_mask_train),
            batch_size=args.batch_size,
            shuffle=True, num_workers=1,
            collate_fn=partial_mv_tabular_collate)

        model = TDR_IMV(num_views, dims, num_classes, device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        model.to(device)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        def pretrain(epoch):
            model.train()
            loss_meter = AverageMeter()
            for batch_idx, (data, sn, target, lmask) in enumerate(train_loader):
                for v_num in range(len(data)):
                    data[v_num] = Variable(data[v_num].float().to(device))
                target = Variable(target.long().to(device))
                sn = Variable(sn.long().to(device))
                lmask = lmask.bool().to(device)
                optimizer.zero_grad()
                if lmask.any():
                    labeled_data   = [d[lmask] for d in data]
                    labeled_target = target[lmask]
                    evidences, evidence_a, evidence_con, evidence_div = model(
                        labeled_data, labeled_target, epoch, beta)
                    loss = get_loss(evidences, evidence_a, evidence_con, evidence_div,
                                    labeled_target, epoch, num_classes,
                                    args.annealing_epochs, gamma, delta, device)
                    loss.backward()
                    optimizer.step()
                    loss_meter.update(loss.item())
            print('Pretrain Epoch: {} \tLoss: {:.6f}'.format(epoch, loss_meter.avg))

        def train(epoch):
            model.train()
            loss_meter = AverageMeter()
            for batch_idx, (data, sn, target, lmask) in enumerate(train_loader):
                for v_num in range(len(data)):
                    data[v_num] = Variable(data[v_num].float().to(device))
                target = Variable(target.long().to(device))
                sn = Variable(sn.long().to(device))
                lmask = lmask.bool().to(device)
                optimizer.zero_grad()
                if lmask.any():
                    labeled_data   = [d[lmask] for d in data]
                    labeled_target = target[lmask]
                    evidences, evidence_a, evidence_con, evidence_div = model(
                        labeled_data, labeled_target, epoch, beta)
                    loss = get_loss(evidences, evidence_a, evidence_con, evidence_div,
                                    labeled_target, epoch, num_classes,
                                    args.annealing_epochs, gamma, delta, device)
                    loss.backward()
                    optimizer.step()
                    loss_meter.update(loss.item())
            return loss_meter.avg

        def test(epoch, dataloader):
            model.eval()
            num_correct, num_sample = 0, 0
            for batch_idx, (data, target) in enumerate(dataloader):
                for v_num in range(len(data)):
                    data[v_num] = Variable(data[v_num].float().to(device))
                target = Variable(target.long().to(device))
                num_sample += target.size(0)
                with torch.no_grad():
                    evidence, evidences, evidence_con, evidence_div = model(data, target, epoch, beta)
                    _, Y_pre = torch.max(evidences, dim=1)
                    num_correct += (Y_pre == target).sum().item()
            return num_correct / num_sample

        # Pre-training
        for epo in range(1, args.preepoch):
            pretrain(epo)

        # Uncertainty filtering on training set
        print("Sampled points filtering...")
        X_train, Y_train, filter_mask = Uncertainty_filter(
            X_train, Y_train, model, num_classes, args.preepoch, beta, args, device)
        Sn_train = Sn_train[filter_mask]
        if label_mask_train is not None:
            label_mask_train = label_mask_train[filter_mask]
        train_loader = DataLoader(
            dataset=partial_mv_dataset(X_train, Sn_train, Y_train, label_mask_train),
            batch_size=args.batch_size,
            shuffle=True, num_workers=1,
            collate_fn=partial_mv_tabular_collate)

        # Main training
        losss = []
        for epoch in range(1, args.epochs + 1):
            loss = train(epoch)
            losss.append(loss)
            early_stopping(loss * (-1), model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            print('Train Epoch: {} Loss: {:.6f}'.format(epoch, loss))

        # Evaluation
        X_filter_test, Y_filter_test, _ = Uncertainty_filter(
            X_test, Y_test, model, num_classes, epoch, beta, args, device)
        test_loader = DataLoader(
            dataset=mv_dataset(X_filter_test, Y_filter_test),
            batch_size=10, shuffle=False, num_workers=1,
            collate_fn=mv_tabular_collate)
        acc = test(1, test_loader)
        all_results.append(acc)
        print(f"Run {run + 1} accuracy: {acc:.4f}")

    mean_acc = float(np.mean(all_results))
    std_acc  = float(np.std(all_results))
    elapsed = time.time() - t_start
    print(f"\n[{dataset_name_key}] n_labeled={n_labeled_per_class}  "
          f"elapsed: {elapsed:.1f}s  mean={mean_acc:.4f}  std={std_acc:.4f}")
    return mean_acc, std_acc, all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=500, metavar='N')
    parser.add_argument('--epochs', type=int, default=5, metavar='N')
    parser.add_argument('--preepoch', type=int, default=10, metavar='N')
    parser.add_argument('--annealing-epochs', type=int, default=20, metavar='N')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR')
    parser.add_argument('--patience', type=int, default=3, metavar='LR')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--missing-rate', type=float, default=0.5, metavar='LR')
    parser.add_argument('--n-sample', type=int, default=30, metavar='LR')
    parser.add_argument('--k', type=int, default=5, metavar='LR')
    parser.add_argument('--annealing_step', type=int, default=50, metavar='N')
    parser.add_argument('--label-seed', type=int, default=41, metavar='N',
                        help='base random seed; run i uses seed label_seed+i (default: 41)')
    parser.add_argument('--n-runs', type=int, default=5, metavar='N',
                        help='number of repeated runs per (dataset, n_labeled) combination (default: 5)')
    parser.add_argument('--datasets', nargs='+', default=['Wikipedia'],#, 'MNIST10k','Caltech101all'
                        help='datasets to evaluate')#['sources3', 'ALOI100', 'leaves100', 'Caltech101all']
    parser.add_argument('--n-labeled-list', nargs='+', type=int, default=[1,2,3,4,5],
                        help='list of n_labeled_per_class values to sweep (default: 1 2 3 4 5)')
    args = parser.parse_args()

    # ── Sweep over datasets × n_labeled ──
    summary = {}   # {(dataset, n_labeled): (mean, std, [runs])}

    for ds_name in args.datasets:
        if ds_name not in DATASET_REGISTRY:
            print(f"[WARNING] Unknown dataset '{ds_name}', skipping.")
            continue
        for n_labeled in args.n_labeled_list:
            mean_acc, std_acc, runs = run_experiment(ds_name, n_labeled, args)
            summary[(ds_name, n_labeled)] = (mean_acc, std_acc, runs)

            # Per-combination result line
            print(f"\n>>> [{ds_name}] n_labeled={n_labeled}: "
                  f"mean={mean_acc:.4f}  std={std_acc:.4f}  runs={[round(r,4) for r in runs]}")

            # Append to per-dataset result file
            result_line = (f"n_labeled:{n_labeled}"
                           f"\tmissing_rate:{args.missing_rate}"
                           f"\truns:{args.n_runs}"
                           f"\tresults:{runs}"
                           f"\tmean:{mean_acc:.4f}"
                           f"\tstd:{std_acc:.4f}\n")
            with open(f"./{ds_name}_results.txt", "a") as f:
                f.write(result_line)

    # ── Final summary table ──
    print(f"\n{'='*70}")
    print(f"{'SUMMARY':^70}")
    print(f"{'='*70}")
    header = f"{'Dataset':<12}" + "".join(f"  n={n:<6}" for n in args.n_labeled_list)
    print(header)
    print("-" * len(header))
    for ds_name in args.datasets:
        row = f"{ds_name:<12}"
        for n_labeled in args.n_labeled_list:
            key = (ds_name, n_labeled)
            if key in summary:
                m, s, _ = summary[key]
                row += f"  {m:.4f}±{s:.4f}"
            else:
                row += f"  {'N/A':>13}"
        print(row)
    print(f"{'='*70}")
