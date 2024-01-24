import torch
import networkx as nx
from sklearn.metrics import precision_score, recall_score


# def hyperbolic(x, y, keepdim=True):
#     res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
#     if keepdim:
#         res = res.view(res.shape + (1,))
#     return res

def hyperbolic(x, y):
    x = x.clone().detach()
    x[:, 0] = -x[:, 0]
    return torch.inner(x, y)

def proj(x):
    K = 1.0
    d = x.size(-1) - 1
    y = x.narrow(-1, 1, d)
    y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2 
    mask = torch.ones_like(x)
    mask[:, 0] = 0
    vals = torch.zeros_like(x)
    vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=1e-7))
    return vals + mask * x

def logmap0(x):
    K = 1.0
    sqrtK = K ** 0.5
    d = x.size(-1) - 1
    y = x.narrow(-1, 1, d).view(-1, d)
    y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
    y_norm = torch.clamp(y_norm, min=1e-7)
    res = torch.zeros_like(x)
    theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + 1e-7)
    res[:, 1:] = sqrtK * torch.arccosh(theta) * y / y_norm
    return res

def expmap0(u):
    K = 1.0
    sqrtK = K ** 0.5
    d = u.size(-1) - 1
    x = u.narrow(-1, 1, d).view(-1, d)
    x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
    x_norm = torch.clamp(x_norm, min=1e-7)
    theta = x_norm / sqrtK
    res = torch.ones_like(u)
    res[:, 0:1] = sqrtK * torch.cosh(theta)
    res[:, 1:] = sqrtK * torch.sinh(theta) * x / x_norm
    return proj(res)

def mobius_matvec(m, x):
    u = logmap0(x)
    mu = u @ m.transpose(-1, -2)
    return expmap0(mu)

def euclidean(x, y, keepdim=True):
    return torch.inner(x, y)

def spherical(x, y, keepdim=True):
    return torch.inner(x, y)

def combined_inner(x, y, slices=(1, 3, 2), dims=(17, 16, 17), keepdim=True):
    spaces = [hyperbolic, euclidean, spherical]
    start, end = 0, dims[0]
    prods = []
    for space_idx, (val, slice_dim) in enumerate(zip(slices, dims)):
        prod = spaces[space_idx]
        end = start + slice_dim
        for i in range(val):
            curr_x, curr_y = x[:, start:end], y[:, start:end]
            start += slice_dim
            end += slice_dim
            prods += [prod(curr_x, curr_y)]
    return prods

def predict(embs, sd=2):
    prods = combined_inner(embs, embs)
    all_preds = []
    for prod in prods:
        abs_prod = torch.abs(prod)
        median = abs_prod.median()
        std = abs_prod.std()
        preds = torch.where(abs_prod > sd * std + median, 1.0, 0.0)
        all_preds.append(preds)
    return all_preds

def average(preds, gt=0.5):
    total = torch.zeros_like(preds[0])
    for pred in preds:
        total = torch.add(total, pred)
    total /= len(preds)
    return total, torch.where(total >= gt, 1.0, 0.0)

def load_emb_data(emb_path, edge_path):
    emb = torch.load(emb_path).detach()
    graph_num = edge_path.split('/')[-1].split('.')[0].split('_')[0]
    pre_path = '/'.join(edge_path.split('/')[:-1])
    string_edge_path = f"{pre_path}/{graph_num}_string_cleaned.edges"
    G1 = nx.read_edgelist(edge_path)
    G2 = nx.read_edgelist(string_edge_path)
    adj1 = nx.adjacency_matrix(G1)
    adj2 = nx.adjacency_matrix(G2)
    return adj1, adj2, emb

if __name__ == "__main__":
    emb_path = 'models/624.pt'
    edge_path = '../data/624/624.edges'
    val_adj, test_adj, embs = load_emb_data(emb_path, edge_path)
    val_adj, test_adj = torch.tensor(val_adj.todense()), torch.tensor(test_adj.todense())
    dim = embs.size(0)
    prods = combined_inner(embs, embs, slices=(3, 1, 2), dims=(16, 16, 16))
    preds = predict(embs, sd=1.2)
    avg, best = average(preds, gt=0.1)
    idx = torch.triu_indices(best.size(0), best.size(1))
    mask = torch.eye(dim, dim).bool()
    best.masked_fill_(mask, 0)
    best_vec = torch.tensor([best[i,j] for i, j in zip(idx[0], idx[1])])
    val_adj_vec = torch.tensor([val_adj[i,j] for i, j in zip(idx[0], idx[1])])
    test_adj_vec = torch.tensor([test_adj[i,j] for i, j in zip(idx[0], idx[1])])
    val_acc = 1.0 - torch.count_nonzero(val_adj_vec - best_vec) / best_vec.size(0)
    test_acc = 1.0 - torch.count_nonzero(test_adj_vec - best_vec) / best_vec.size(0)
    val_precision = precision_score(test_adj_vec, best_vec)
    val_recall = recall_score(test_adj_vec, best_vec)
    test_precision = precision_score(test_adj_vec, best_vec)
    test_recall = recall_score(test_adj_vec, best_vec)

