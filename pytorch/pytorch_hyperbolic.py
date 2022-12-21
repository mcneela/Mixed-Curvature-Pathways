import logging
import argparse
import os, sys
import networkx as nx
import random
import wandb

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset


import matplotlib as mpl
if torch.cuda.is_available(): mpl.use('Agg')
import matplotlib.pyplot as plt
if torch.cuda.is_available(): plt.ioff()
import scipy
import scipy.sparse.csgraph as csg
import pandas
import numpy as np, math

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
import utils.load_graph as load_graph
import utils.vis as vis
import utils.distortions as dis
import graph_helpers as gh
import mds_warmstart
from hyperbolic_models import ProductEmbedding
from hyperbolic_parameter import RParameter


# This describes a hyperbolic optimizer in Pytorch. It requires two modifications:
#
# * When declaring a parameter, one uses a class called "Hyperbolic
# * Parameter". It assumes that the _last_ dimension is in the
# * disk. E.g., a tensor of size n x m x d means that you have n x m
# * elements of H_D. d >= 2.
#
#   * It inherits from parameter, so you can do anything you want with it (set its value, pass it around).
#
# * So that you can use any optimizer and get the correction, after the `backward` call but before the `step`, you need to call a function called `hyperbolic_fix`. It will walk the tree, look for the hyperbolic parameters and correct them.
#  * The step function below can be used pretty generically.



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

def cu_var(x):
    if isinstance(x, list) : return [cu_var(u) for u in x]
    if isinstance(x, tuple): return tuple([cu_var(u) for u in list(x)])
    return torch.tensor(x, device=device)

def unwrap(x):
    """ Extract the numbers from (sequences of) pytorch tensors """
    if isinstance(x, list) : return [unwrap(u) for u in x]
    if isinstance(x, tuple): return tuple([unwrap(u) for u in list(x)])
    return x.detach().cpu().numpy()

#
# Dataset extractors
#

class GraphRowSubSampler(torch.utils.data.Dataset):
    def __init__(self, G, scale, subsample, weight_fn, Z=None):
        super(GraphRowSubSampler, self).__init__()
        self.graph     = nx.to_scipy_sparse_matrix(G, nodelist=list(range(G.order())))
        self.n         = G.order()
        self.scale     = scale
        self.subsample = subsample if subsample > 0 else self.n-1
        self.val_cache = torch.zeros((self.n,self.subsample), dtype=torch.double)
        self.idx_cache = torch.LongTensor(self.n,self.subsample,2).zero_()
        self.w_cache   = torch.zeros((self.n, self.subsample), dtype=torch.double)
        self.cache     = set()
        self.verbose   = False
        self.n_cached  = 0
        self.Z         = Z
        self.nbr_frac  = 1.0 # fill up this proportion of samples with neighbors
        self.weight_fn = weight_fn
        logging.info(self)

        ## initialize up front
        for i in range(self.n):
            self.__getitem__(i)
        ## store total weight
        self.total_w = torch.sum(self.w_cache)
        self.max_dist = torch.max(self.val_cache)


    def __getitem__(self, index):
        if index not in self.cache:
            if self.verbose: logging.info(f"Cache miss for {index}")
            h = gh.djikstra_wrapper( (self.graph, [index]) )[0,:] if self.Z is None else self.Z[index,:]
            # add in all the edges
            cur = 0
            self.idx_cache[index,:,0] = index
            neighbors = scipy.sparse.find(self.graph[index,:])[1]
            for e in neighbors:
                self.idx_cache[index,cur,1] = int(e)
                self.val_cache[index,cur] = self.scale*h[e]
                self.w_cache[index,cur] = self.weight_fn(1.0)
                cur += 1
                if cur >= self.nbr_frac * self.subsample: break

            scratch   = np.array(range(self.n))
            np.random.shuffle(scratch)

            i = 0
            while cur < self.subsample and i < self.n:
                v = scratch[i]
                if v != index and v not in neighbors:
                    self.idx_cache[index,cur,1] = int(v)
                    self.val_cache[index,cur]   = self.scale*h[v]
                    # self.val_cache[index,cur]   = 0
                    self.w_cache[index,cur] = self.weight_fn(h[v])
                    cur += 1
                i += 1
            if self.verbose: logging.info(f"\t neighbors={neighbors} {self.idx_cache[index,:,1].numpy().T}")
            self.cache.add(index)
            self.n_cached += 1
            # if self.n_cached % (max(self.n//20,1)) == 0: logging.info(f"\t Cached {self.n_cached} of {self.n}")

        # print("GraphRowSubSampler: idx shape ", self.idx_cache[index,:].size())
        return (self.idx_cache[index,:], self.val_cache[index,:], self.w_cache[index,:])

    def __len__(self): return self.n

    def __repr__(self):
        return f"Subsample: {self.n} points with scale {self.scale} subsample={self.subsample}"


class GraphRowSampler(torch.utils.data.Dataset):
    def __init__(self, G, scale, use_cache=True):
        self.graph = nx.to_scipy_sparse_matrix(G, nodelist=list(range(G.order())))
        self.n     = G.order()
        self.scale = scale
        self.cache = dict() if use_cache else None

    def __getitem__(self, index):
        h = None
        if self.cache is None or index not in self.cache:
            h = gh.djikstra_wrapper( (self.graph, [index]) )
            if self.cache is not None:
                self.cache[index] = h
            #logging.info(f"info {index}")
        else:
            h = self.cache[index]
            #logging.info(f"hit {index}")

        idx = torch.tensor([ (index, j) for j in range(self.n) if j != index], dtype=torch.long)
        v   = torch.DoubleTensor(h).view(-1)[idx[:,1]]
        return (idx, v)

    def __len__(self): return self.n

    def __repr__(self):
        return f"DATA: {self.n} points with scale {self.scale}"


def collate(ls):
    x, y = zip(*ls)
    return torch.cat(x), torch.cat(y)
def collate3(ls):
    x, y, z = zip(*ls)
    return torch.cat(x), torch.cat(y), torch.cat(z)

def build_dataset(G, lazy_generation, sample, subsample, scale, batch_size, weight_fn, num_workers):
    n = G.order()
    Z = None

    logging.info("Building dataset")

    if subsample is not None and (subsample <= 0 or subsample >= n):
        subsample = n-1

    if lazy_generation:
        if subsample is not None:
            z = DataLoader(GraphRowSubSampler(G, scale, subsample, weight_fn), batch_size//subsample, shuffle=True, collate_fn=collate3)
        else:
            z = DataLoader(GraphRowSampler(G, scale), batch_size//(n-1), shuffle=True, collate_fn=collate)
        logging.info("Built Data Sampler")
    else:
        Z   = gh.build_distance(G, scale, num_workers=int(num_workers) if num_workers is not None else 16)   # load the whole matrix
        logging.info(f"Built distance matrix with {scale} factor")

        if subsample is not None:
            z = DataLoader(GraphRowSubSampler(G, scale, subsample, weight_fn, Z=Z), batch_size//subsample, shuffle=True, collate_fn=collate3)
        else:
            idx       = torch.LongTensor([(i,j)  for i in range(n) for j in range(i+1,n)])
            Z_sampled = gh.dist_sample_rebuild_pos_neg(Z, sample) if sample < 1 else Z
            vals      = torch.DoubleTensor([Z_sampled[i,j] for i in range(n) for j in range(i+1, n)])
            z         = DataLoader(TensorDataset(idx,vals), batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
            # TODO does this not shuffle immediately?
        logging.info("Built data loader")

    return Z, z


#
# DATA Diagnostics
#
def major_stats(G, n, m, lazy_generation, Z,z, fig, ax, writer, visualize, subsample, n_rows_sampled=256, num_workers=16):
    m.train(False)
    if lazy_generation:
        logging.info(f"\t Computing Major Stats lazily... ")
        avg, me, mc = 0.0, 0.0, 0.0
        good,bad    = 0,0
        _count      = 0
        for u in z:
            index,vs,_ = u
            v_rec  = unwrap(m.dist_idx(index.to(device)))
            v      = vs.cpu().numpy()
            for i in range(len(v)):
                if dis.entry_is_good(v[i], v_rec[i]):
                    (_avg,me,mc) = dis.distortion_entry(v[i], v_rec[i], me, mc)
                    avg         += _avg
                    good        += 1
                else:
                    bad         += 1
            _count += len(v)
            # if n_rows_sampled*(n-1) <= _count:
            ss = subsample if subsample is not None else n-1
            if _count >= n_rows_sampled*ss:
                break
        logging.info(f"\t\t Completed edges={_count} good={good} bad={bad}")

        avg_dist     = avg/good if good > 0 else 0
        wc_dist      = me*mc
        nan_elements = bad
        map_avg      = 0.0

        # sample for rows
        shuffled     = list(range(n))
        np.random.shuffle(shuffled)
        mm = 0
        for i in shuffled[0:n_rows_sampled]:
            h_rec      = unwrap(m.dist_row(i))
            map_avg   += dis.map_via_edges(G,i, h_rec)
            mm        += 1
        mapscore = map_avg/mm
    else:
        H    = Z
        Hrec = unwrap(m.dist_matrix())
        logging.info("Compare matrices built")
        mc, me, avg_dist, nan_elements = dis.distortion(H, Hrec, n, num_workers)
        wc_dist = me*mc
        mapscore = dis.map_score(scipy.sparse.csr_matrix.todense(G).A, Hrec, n, num_workers)

    if visualize:
        num_spheres = np.minimum(len(m.S), 5)
        num_hypers  = np.minimum(len(m.H), 5)
        for emb in range(num_spheres):
            ax_this = vis.get_ax(num_hypers, num_spheres, ax, emb, 1)
            ax_this.cla()
        for emb in range(num_hypers):
            ax_this = vis.get_ax(num_hypers, num_spheres, ax, emb, 0)
            ax_this.cla()

        text_3d_only = False

        vis.draw_graph(G,m,fig, ax)
        if num_hypers > 0:
            axlabel = vis.get_ax(num_hypers, num_spheres, ax, 0, 0)
        else:
            axlabel = vis.get_ax(num_hypers, num_spheres, ax, 0, 1)
            sdim = 0 if len(m.S) == 0 else len((m.S[0]).w[0])
            if sdim == 3: text_3d_only = True

        if text_3d_only:
            axlabel.text(-1.00, 1.0, 1.1, "Epoch "+str(m.epoch), fontsize=20)
            axlabel.text(-1.00, 1.0, 0.8, "MAP "+str(mapscore)[0:5], fontsize=20)
        else:
            axlabel.text(0.70, 1.1, "Epoch "+str(m.epoch), fontsize=20)
            axlabel.text(0.70, 1.0, "MAP "+str(mapscore)[0:5], fontsize=20)

        writer.grab_frame()

    logging.info(f"Distortion avg={avg_dist} wc={wc_dist} me={me} mc={mc} nan_elements={nan_elements}")
    logging.info(f"MAP = {mapscore}")
    logging.info(f"scale={unwrap(m.scale())}")

    return avg_dist, wc_dist, me, mc, mapscore

def learn(dataset, dim, hyp, edim, euc, sdim, sph, scale, riemann, learning_rate, decay_length, decay_step, momentum, tol, epochs, burn_in,
          use_yellowfin, use_adagrad, resample_freq, print_freq, model_save_file, model_load_file, batch_size,
          num_workers, lazy_generation, log_name, log, warm_start, learn_scale, checkpoint_freq, sample, subsample,
          logloss, distloss, squareloss, symloss, exponential_rescale, extra_steps, use_svrg, T, use_hmds, visualize, table=None, artifact=None):
    # Log configuration
    if log_name is None and log:
        if not os.path.exists('log'):
            os.makedirs('log')
        log_name = f"log/{dataset.split('/')[1]}.H{dim}-{hyp}.E{edim}-{euc}.S{sdim}-{sph}.lr{learning_rate}.log"
    formatter = logging.Formatter('%(asctime)s %(message)s')
    logging.basicConfig(filename=log_name,
                        level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%FT%T',)
    if log_name is not None:
        logging.info(f"Logging to {log_name}")
        log = logging.getLogger()
        fh  = logging.FileHandler(log_name)
        fh.setFormatter( formatter )
        log.addHandler(fh)

    logging.info(f"Commandline {sys.argv}")
    if model_save_file is None: logging.warn("No Model Save selected!")
    G  = load_graph.load_graph(dataset)
    GM = nx.to_scipy_sparse_matrix(G, nodelist=list(range(G.order())))

    # grab scale if warm starting:
    if warm_start:
        scale = pandas.read_csv(warm_start, index_col=0).as_matrix()[0, -1]

    n = G.order()
    logging.info(f"Loaded Graph {dataset} with {n} nodes scale={scale}")

    if exponential_rescale is not None:
        # torch.exp(exponential_rescale * -d)
        def weight_fn(d):
            if d <= 2.0: return 5.0
            elif d > 4.0: return 0.01
            else: return 1.0
    else:
        def weight_fn(d):
            return 1.0
    Z, z = build_dataset(G, lazy_generation, sample, subsample, scale, batch_size, weight_fn, num_workers)

    if model_load_file is not None:
        logging.info(f"Loading {model_load_file}...")
        m = torch.load(model_load_file).to(device)
        logging.info(f"Loaded scale {unwrap(m.scale())} {torch.sum(m.embedding().data)} {m.epoch}")
    else:
        logging.info(f"Creating a fresh model warm_start?={warm_start}")

        m_init = None
        if warm_start:
            # load from DataFrame; assume that the julia combinatorial embedding has been saved
            ws_data = pandas.read_csv(warm_start, index_col=0).as_matrix()
            scale = ws_data[0, ws_data.shape[1]-1]
            m_init = torch.DoubleTensor(ws_data[:,range(ws_data.shape[1]-1)])
        elif use_hmds:
            # m_init = torch.DoubleTensor(mds_warmstart.get_normalized_hyperbolic(mds_warmstart.get_model(dataset,dim,scale)[1]))
            m_init = torch.DoubleTensor(mds_warmstart.get_model(dataset,dim,scale)[1])

        logging.info(f"\t Warmstarting? {warm_start} {m_init.size() if warm_start else None} {G.order()}")
        initial_scale = z.dataset.max_dist / 3.0
        print("MAX DISTANCE", z.dataset.max_dist)
        print("AVG DISTANCE", torch.mean(z.dataset.val_cache))
        initial_scale=0.0
        m       = ProductEmbedding(G.order(), dim, hyp, edim, euc, sdim, sph, initialize=m_init, learn_scale=learn_scale, initial_scale=initial_scale, logrel_loss=logloss, dist_loss=distloss, square_loss=squareloss, sym_loss=symloss, exponential_rescale=exponential_rescale, riemann=riemann).to(device)
        m.normalize()
        m.epoch = 0
    logging.info(f"Constructed model with dim={dim} and epochs={m.epoch} isnan={np.any(np.isnan(m.embedding().cpu().data.numpy()))}")

    if visualize:
        name = 'animations/' + f"{os.path.split(os.path.splitext(dataset)[0])[1]}.H{dim}-{hyp}.E{edim}-{euc}.S{sdim}-{sph}.lr{learning_rate}.ep{epochs}.seed{seed}"
        fig, ax, writer = vis.setup_plot(m=m, name=name, draw_circle=True)
    else:
        fig = None
        ax = None
        writer = None

    #
    # Build the Optimizer
    #
    # TODO: Redo this in a sensible way!!

    # per-parameter learning rates
    exp_params = [p for p in m.embed_params if p.use_exp]
    learn_params = [p for p in m.embed_params if not p.use_exp]
    hyp_params = [p for p in m.hyp_params if not p.use_exp]
    euc_params = [p for p in m.euc_params if not p.use_exp]
    sph_params = [p for p in m.sph_params if not p.use_exp]
    scale_params = m.scale_params
    model_params = [
        {'params': hyp_params},
        {'params': euc_params},
        {'params': sph_params,
         'lr': 0.1*learning_rate},
        {'params': m.scale_params,
         'lr': 1e-4*learning_rate}
    ]

    # opt = None
    if len(model_params) > 0:
        opt = torch.optim.SGD(model_params, lr=learning_rate/10, momentum=momentum)
    if len(scale_params) > 0:
        scale_opt = torch.optim.SGD(scale_params, lr=1e-3*learning_rate)
        scale_decay = torch.optim.lr_scheduler.StepLR(scale_opt, step_size=1, gamma=.99)
    else:
        scale_opt = None
        scale_decay = None
    lr_burn_in = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[burn_in], gamma=10)
    if use_yellowfin:
        from yellowfin import YFOptimizer
        opt = YFOptimizer(model_params)

    if use_adagrad:
        opt = torch.optim.Adagrad(model_params)

    if use_svrg:
        from svrg import SVRG
        base_opt = torch.optim.Adagrad if use_adagrad else torch.optim.SGD
        opt      = SVRG(m.parameters(), lr=learning_rate, T=T, data_loader=z, opt=base_opt)
        # TODO add ability for SVRG to take parameter groups


    logging.info(opt)

    # Log stats from import: when warmstarting, check that it matches Julia's stats
    logging.info(f"*** Initial Checkpoint. Computing Stats")
    major_stats(GM, n, m, lazy_generation, Z, z, fig, ax, writer, visualize, subsample)
    logging.info("*** End Initial Checkpoint\n")

    # track best stats
    best_loss   = 1.0e10
    best_dist   = 1.0e10
    best_wcdist = 1.0e10
    best_map    = 0.0
    for i in range(m.epoch+1, m.epoch+epochs+1):
        lr_burn_in.step()

        l, n_edges = 0.0, 0.0 # track average loss per edge
        m.train(True)
        if use_svrg:
            for data in z:
                def closure(data=data, target=None):
                    _data = data if target is None else (data,target)
                    c = m.loss(_data.to(device))
                    c.backward()
                    return c.data[0]
                l += opt.step(closure)

                # Projection
                m.normalize()

        else:
            for the_step in range(extra_steps):
                # Accumulate the gradient
                for u in z:
                    # Zero out the gradients
                    # if opt is not None: opt.zero_grad() # This is handled by the SVRG.
                    # if exp is not None: exp.zero_grad()
                    opt.zero_grad()
                    for p in exp_params:
                        if p.grad is not None:
                            p.grad.detach_()
                            p.grad.zero_()
                    # Compute loss
                    _loss = m.loss(cu_var(u))
                    _loss.backward()
                    l += _loss.item() * u[0].size(0)
                    n_edges += u[0].size(0)
                    # modify gradients if necessary
                    RParameter.correct_metric(m.parameters())
                    # step
                    opt.step()
                    for p in exp_params:
                        lr = opt.param_groups[0]['lr']
                        p.exp(lr)
                    # Projection
                    m.normalize()

        l /= n_edges

        # m.epoch refers to num of training epochs finished
        m.epoch += 1

        if i % print_freq == 0:
            logging.info(f"{i} loss={l}")
        if (i <= burn_in and i % (checkpoint_freq/5) == 0) or i % checkpoint_freq == 0:
            logging.info(f"\n*** Major Checkpoint. Computing Stats and Saving")
            avg_dist, wc_dist, me, mc, mapscore = major_stats(GM,n,m, True, Z, z, fig, ax, writer, visualize, subsample)
            #wandb.log({
            #    'Average Distortion': avg_dist,
            #    'Worst-Case Distortion': wc_dist,
            #    'Maximum Expansion': me,
            #    'Maximum Compression': mc
            #})
            best_loss   = min(best_loss, l)
            best_dist   = min(best_dist, avg_dist)
            best_wcdist = min(best_wcdist, wc_dist)
            best_map    = max(best_map, mapscore)
            if model_save_file is not None:
                fname = f"{model_save_file}.{m.epoch}"
                logging.info(f"Saving model into {fname} {torch.sum(m.embedding().data)} ")
                torch.save(m, fname)
            logging.info("*** End Major Checkpoint\n")
        if i % resample_freq == 0:
            if sample < 1. or subsample is not None:
                Z, z = build_dataset(G, lazy_generation, sample, subsample, scale, batch_size, weight_fn, num_workers)

    logging.info(f"final loss={l}")
    logging.info(f"best loss={best_loss}, distortion={best_dist}, map={best_map}, wc_dist={best_wcdist}")
    # wandb.log({
    #     'Final Loss': l,
    #     'Best Loss': best_loss,
    #     'Best Distortion': best_dist,
    #     'Best MAP': best_map,
    #     'Best WC Distortion': best_wcdist
    # })
    #table.add_data(l, best_loss, best_dist, best_map, best_wcdist)
    #artifact.add( table, "predictions")
    #wandb.run.log_artifact(artifact)


    final_dist, final_wc, final_me, final_mc, final_map = major_stats(GM, n, m, lazy_generation, Z,z, fig, ax, writer, False, subsample)


    if log_name is not None:
        with open(log_name + '.stat', "w") as f:
            f.write("Best-loss MAP dist wc Final-loss MAP dist wc me mc\n")
            f.write(f"{best_loss:10.6f} {best_map:8.4f} {best_dist:8.4f} {best_wcdist:8.4f} {l:10.6f} {final_map:8.4f} {final_dist:8.4f} {final_wc:8.4f} {final_me:8.4f} {final_mc:8.4f}")

    if visualize:
        writer.finish()

    if model_save_file is not None:
        fname = f"{model_save_file}.final"
        logging.info(f"Saving model into {fname}-final {torch.sum(m.embedding().data)} {unwrap(m.scale())}")
        torch.save(m, fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset number")
    # model params
    parser.add_argument("-d", "--dim", type=int, default=33, help="Dimension to use")
    parser.add_argument("--hyp", type=int, default=1, help="Number of copies of hyperbolic space")
    parser.add_argument("--edim", type=int, default=33, help="Euclidean dimension to use")
    parser.add_argument("--euc", type=int, default=1, help="Number of copies of Euclidean space")
    parser.add_argument("--sdim", type=int, default=33, help="Spherical dimension to use")
    parser.add_argument("--sph", type=int, default=1, help="Number of copies of spherical space")
    parser.add_argument("--riemann", default=False, action="store_true", help="Use Riemannian metric for product space. Otherwise, use L1 sum")
    parser.add_argument("-s", "--scale", type=float, default=1.0, help="Scale factor")
    parser.add_argument("-t", "--tol", type=float, default=1e-8, help="Tolerances for projection")
    # optimizers and params
    parser.add_argument("-y", "--use-yellowfin", action="store_true", default=False, help="Turn off yellowfin")
    parser.add_argument("--use-adagrad", action="store_true", default=False, help="Use adagrad")
    parser.add_argument("--use-svrg", action="store_true", default=False, help="Use SVRG")
    parser.add_argument("-T", type=int, default=10, help="SVRG T parameter")
    parser.add_argument("--use-hmds", action="store_true", default=False, help="Use MDS warmstart")
    parser.add_argument("-l", "--learning-rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--decay-length", type=int, default=1000, help="Number of epochs per lr decay")
    parser.add_argument("--decay-step", type=int, default=1, help="Size of lr decay")
    parser.add_argument("--momentum", type=float, default=0.0, help="Momentum")
    parser.add_argument("--epochs", type=int, default=100, help="number of steps in optimization")
    parser.add_argument("--burn-in", type=int, default=0, help="number of epochs to initially train at lower LR")
    parser.add_argument("-x", "--extra-steps", type=int, default=1, help="Steps per batch")
    # data
    parser.add_argument("--num-workers", type=int, default=None, help="Number of workers for loading. Default is to use all cores")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (number of edges)")
    parser.add_argument("--sample", type=float, default=1.0, help="Sample the distance matrix")
    parser.add_argument("-g", "--lazy-generation", action="store_true", default=False, help="Use a lazy data generation technique")
    parser.add_argument("--subsample", type=int, default=None, help="Number of edges per row to subsample")
    parser.add_argument("--resample-freq", type=int, default=1000, help="Resample data frequency (expensive)")
    # logging and saving
    parser.add_argument("--print-freq", type=int, default=1, help="Print loss this every this number of steps")
    parser.add_argument("--checkpoint-freq", type=int, default=100, help="Checkpoint Frequency (Expensive)")
    parser.add_argument("--model-save-file", type=str, default=None, help="Save model file")
    parser.add_argument("--model-load-file", type=str, default=None, help="Load model file")
    parser.add_argument("-w", "--warm-start", action="store_true", default=False, help="Warm start the model with MDS")
    parser.add_argument("--log-name", type=str, default=None, help="Log to a file")
    parser.add_argument("--log", action="store_true", default=False, help="Log to a file (automatic name)")
    # misc
    parser.add_argument("--learn-scale", action="store_true", default=False, help="Learn scale")
    parser.add_argument("--logloss", action="store_true", default=False)
    parser.add_argument("--distloss", action="store_true", default=False)
    parser.add_argument("--squareloss", action="store_true", default=False)
    parser.add_argument("--symloss", action="store_true", default=False)
    parser.add_argument("-e", "--exponential-rescale", type=float, default=None, help="Exponential Rescale")
    parser.add_argument("--visualize", action="store_true", default=False, help="Produce an animation (dimension 2 only)")

    args = parser.parse_args()
    #wandb.init(project="mixed-curvature", entity="mcneela", config=args)
    # wandb.log(**vars(args))
    #wandb.run.name = args.dataset.split('/')[1]
    #config = wandb.config

    columns = ['Final Loss', 'Best Loss', 'Best Distortion', \
               'Best MAP', 'Best WC Distortion']
    #table = wandb.Table(columns=columns)
    #run_data = wandb.Artifact("run_data_" + str(wandb.run.id), type="predictions")

    args = vars(args)
    args = {k: v for k, v in args.items()}
    #learn(**args, table=table, artifact=run_data)
    learn(**args)
