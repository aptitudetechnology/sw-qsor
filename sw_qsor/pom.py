"""OpenPOM's MPNN ("principal odour map") reimplemented in plain PyTorch, CPU, deterministic.

Ported for plan-milestone-4.md from the code it mirrors, read at these sources:
  - OpenPOM (BioMachineLearning/openpom @ 5434c4b): feat/graph_featurizer.py,
    models/mpnn_pom.py, layers/pom_mpnn_gnn.py, layers/pom_ffn.py, utils/loss.py,
    utils/data_utils.py, examples/ensemble_cv.ipynb (the published 5-fold ensemble).
  - DeepChem (master, as OpenPOM calls it): utils/molecule_feature_utils.py,
    models/torch_models/torch_model.py (_StandardLoss, fit loop, padded batches),
    models/optimizers.py (Adam, ExponentialDecay).
  - DGL / dgl-lifesci (master): NNConv, Set2Set, MPNNGNN.
OpenPOM is MIT licensed. Nothing here imports it, DeepChem or DGL.

Faithful to the code, not the docstrings, in two places worth knowing:
  - Label weights are log(1 + count/max_count), so COMMON labels weigh more; OpenPOM's
    loss docstring says the opposite.
  - The loss feeds the raw logit l, as if it were a probability, into a 2-class softmax over
    [1 - l, l]. That equals binary cross-entropy on the logit 2l - 1. Prediction is
    sigmoid(l). This is what the published model trained on.
One computational difference, numerically equivalent: DGL recomputes NNConv's edge network
on every message-passing step from the same edge features; here it is computed once per
forward pass.

Single-atom molecules (e.g. hydrogen sulfide as 'S') have no bonds, so the readout, which sums
messages arriving over bonds, gives them an all-zero input (DGL fills zeros for nodes that
receive nothing). Every such molecule therefore gets the same prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

# ---------------------------------------------------------------- featuriser (OpenPOM)

_VALENCE = [0, 1, 2, 3, 4, 5, 6]
_DEGREE = [0, 1, 2, 3, 4, 5]
_NUM_HS = [0, 1, 2, 3, 4]
_CHARGE = [-1, -2, 1, 2, 0]
_ATOMIC = list(range(100))
_HYBRID = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
ATOM_FDIM = sum(len(s) + 1 for s in (_VALENCE, _DEGREE, _NUM_HS, _CHARGE, _ATOMIC)) + len(_HYBRID) + 1
BOND_FDIM = 6


def _one_hot(val, allowable) -> list[float]:
    """DeepChem one_hot_encode with include_unknown_set=True."""
    out = [0.0] * (len(allowable) + 1)
    out[allowable.index(val) if val in allowable else -1] = 1.0
    return out


def atom_features(atom) -> list[float]:
    return (_one_hot(atom.GetTotalValence(), _VALENCE)
            + _one_hot(atom.GetTotalDegree(), _DEGREE)
            + _one_hot(atom.GetTotalNumHs(), _NUM_HS)
            + _one_hot(atom.GetFormalCharge(), _CHARGE)
            + _one_hot(atom.GetAtomicNum() - 1, _ATOMIC)
            + _one_hot(str(atom.GetHybridization()), _HYBRID))


def bond_features(bond) -> list[float]:
    bt = bond.GetBondType()
    return [0.0, float(bt == Chem.BondType.SINGLE), float(bt == Chem.BondType.DOUBLE),
            float(bt == Chem.BondType.TRIPLE), float(bt == Chem.BondType.AROMATIC),
            float(bond.IsInRing())]


@dataclass(frozen=True)
class Graph:
    x: np.ndarray  # (atoms, 134)
    edge_index: np.ndarray  # (2, 2*bonds), both directions, OpenPOM order
    edge_attr: np.ndarray  # (2*bonds, 6)


def featurize(smiles: str) -> Graph:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"unparseable SMILES {smiles!r}")
    x = np.array([atom_features(a) for a in m.GetAtoms()], dtype=np.float32)
    src, dst, ea = [], [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        src += [i, j]
        dst += [j, i]
        ea += [f, f]
    return Graph(x=x, edge_index=np.array([src, dst], dtype=np.int64).reshape(2, -1),
                 edge_attr=np.array(ea, dtype=np.float32).reshape(-1, BOND_FDIM))


@dataclass
class Batch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    graph_of: torch.Tensor  # (atoms,) graph index
    n_graphs: int


def collate(graphs: list[Graph]) -> Batch:
    xs, eis, eas, gid = [], [], [], []
    off = 0
    for g, gr in enumerate(graphs):
        xs.append(gr.x)
        eis.append(gr.edge_index + off)
        eas.append(gr.edge_attr)
        gid.append(np.full(len(gr.x), g))
        off += len(gr.x)
    return Batch(x=torch.from_numpy(np.concatenate(xs)),
                 edge_index=torch.from_numpy(np.concatenate(eis, axis=1)),
                 edge_attr=torch.from_numpy(np.concatenate(eas)),
                 graph_of=torch.from_numpy(np.concatenate(gid)), n_graphs=len(graphs))


# ---------------------------------------------------------------- network

class NNConvSum(nn.Module):
    """DGL NNConv, sum aggregator, residual identity, bias initialised to zero."""

    def __init__(self, dim: int, edge_net: nn.Module):
        super().__init__()
        self.dim = dim
        self.edge_func = edge_net
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, h, edge_index, W):
        src, dst = edge_index
        msg = torch.bmm(h[src].unsqueeze(1), W).squeeze(1)  # (E, dim)
        agg = torch.zeros_like(h).index_add_(0, dst, msg)
        return agg + h + self.bias


class MPNN(nn.Module):
    """dgllife MPNNGNN with OpenPOM's residual NNConv."""

    def __init__(self, node_in=ATOM_FDIM, edge_in=BOND_FDIM, dim=100, edge_hidden=75, steps=5):
        super().__init__()
        self.project_node_feats = nn.Sequential(nn.Linear(node_in, dim), nn.ReLU())
        self.steps = steps
        self.dim = dim
        edge_net = nn.Sequential(nn.Linear(edge_in, edge_hidden), nn.ReLU(),
                                 nn.Linear(edge_hidden, dim * dim))
        self.gnn_layer = NNConvSum(dim, edge_net)
        self.gru = nn.GRU(dim, dim)

    def forward(self, x, edge_index, edge_attr):
        h = self.project_node_feats(x)
        hidden = h.unsqueeze(0)
        W = self.gnn_layer.edge_func(edge_attr).view(-1, self.dim, self.dim)
        for _ in range(self.steps):
            h = F.relu(self.gnn_layer(h, edge_index, W))
            h, hidden = self.gru(h.unsqueeze(0), hidden)
            h = h.squeeze(0)
        return h


def _segment_softmax(e: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    mx = torch.full((n,), -torch.inf, dtype=e.dtype).scatter_reduce(0, seg, e, "amax")
    z = torch.exp(e - mx[seg])
    return z / torch.zeros(n, dtype=e.dtype).index_add_(0, seg, z)[seg]


class Set2Set(nn.Module):
    """DGL Set2Set."""

    def __init__(self, input_dim: int, n_iters: int, n_layers: int):
        super().__init__()
        self.input_dim, self.n_iters, self.n_layers = input_dim, n_iters, n_layers
        self.lstm = nn.LSTM(2 * input_dim, input_dim, n_layers)

    def forward(self, feat, seg, n):
        h = (feat.new_zeros(self.n_layers, n, self.input_dim),
             feat.new_zeros(self.n_layers, n, self.input_dim))
        q_star = feat.new_zeros(n, 2 * self.input_dim)
        for _ in range(self.n_iters):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(n, self.input_dim)
            e = (feat * q[seg]).sum(-1)
            alpha = _segment_softmax(e, seg, n)
            readout = torch.zeros(n, self.input_dim, dtype=feat.dtype).index_add_(
                0, seg, feat * alpha.unsqueeze(-1))
            q_star = torch.cat([q, readout], dim=-1)
        return q_star


class FFN(nn.Module):
    """OpenPOM CustomPositionwiseFeedForward: batch norm on, dropout_at_input_no_act False."""

    def __init__(self, d_input, d_hidden_list, d_output, dropout_p):
        super().__init__()
        dims = [d_input] + list(d_hidden_list)
        self.linears = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]
                                     + [nn.Linear(dims[-1], d_output)])
        self.batchnorms = nn.ModuleList([nn.BatchNorm1d(d) for d in d_hidden_list])
        self.dropout = nn.Dropout(dropout_p)
        self.n_layers = len(d_hidden_list) + 1

    def forward(self, x):
        for i in range(self.n_layers - 2):
            x = self.dropout(F.relu(self.batchnorms[i](self.linears[i](x))))
        embeddings = self.linears[self.n_layers - 2](x)
        x = self.dropout(F.relu(self.batchnorms[self.n_layers - 2](embeddings)))
        return embeddings, self.linears[-1](x)


class POM(nn.Module):
    """OpenPOM MPNNPOM with the published settings (examples/ensemble_cv.ipynb)."""

    def __init__(self, n_tasks: int, node_out=100, edge_hidden=75, edge_out=100, steps=5,
                 s2s_steps=3, s2s_layers=2, ffn_hidden=(392, 392), ffn_emb=256, dropout=0.12):
        super().__init__()
        self.mpnn = MPNN(dim=node_out, edge_hidden=edge_hidden, steps=steps)
        self.project_edge_feats = nn.Sequential(nn.Linear(BOND_FDIM, edge_out), nn.ReLU())
        self.readout_set2set = Set2Set(node_out + edge_out, s2s_steps, s2s_layers)
        self.ffn = FFN(2 * (node_out + edge_out), list(ffn_hidden) + [ffn_emb], n_tasks, dropout)

    def forward(self, b: Batch):
        h = self.mpnn(b.x, b.edge_index, b.edge_attr)
        src, dst = b.edge_index
        msg = torch.cat([h[src], self.project_edge_feats(b.edge_attr)], dim=1)
        node = torch.zeros(len(h), msg.shape[1], dtype=h.dtype).index_add_(0, dst, msg)
        mol = self.readout_set2set(node, b.graph_of, b.n_graphs)
        emb, logits = self.ffn(mol)
        return logits, emb  # logits (batch, tasks); prediction = sigmoid(logits)


# ---------------------------------------------------------------- loss and training

def imbalance_weights(Y: np.ndarray) -> torch.Tensor:
    """log(1 + count / max_count) per label, on the training labels (OpenPOM code)."""
    c = np.asarray(Y).sum(0).astype(np.float64)
    return torch.log(1 + torch.tensor(c / c.max(), dtype=torch.float32))


def pom_loss(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """CustomMultiLabelLoss ('sum') through DeepChem's _StandardLoss: per-sample sum over labels
    of w_t * CE(softmax([1 - l, l]), y), then the mean over the batch."""
    two = torch.stack([1 - logits, logits], dim=1)  # (batch, 2, tasks)
    ce = F.cross_entropy(two, y.long(), reduction="none")  # (batch, tasks)
    return (ce * w).sum(1).mean()


def regularization(model: nn.Module, weight_decay: float) -> torch.Tensor:
    """MPNNPOMModel._regularization_loss: wd * (sum of L1 norms + sum of L2 norms), excluding
    parameters whose name contains 'bias'."""
    l1 = sum(p.abs().sum() for n, p in model.named_parameters() if "bias" not in n)
    l2 = sum(p.norm(2) for n, p in model.named_parameters() if "bias" not in n)
    return weight_decay * (l1 + l2)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 62
    batch_size: int = 128
    lr: float = 1e-3
    decay_rate: float = 0.5
    decay_steps: int = 640  # 32 * 20, staircase, stepped per batch
    weight_decay: float = 1e-5
    threads: int = 12


def _batches(n: int, batch_size: int, rng: np.random.Generator):
    """One shuffled epoch; the last batch padded to full size by cycling its own samples
    (DeepChem pad_batches=True)."""
    perm = rng.permutation(n)
    for s in range(0, n, batch_size):
        idx = perm[s:s + batch_size]
        if len(idx) < batch_size:
            idx = np.resize(idx, batch_size)
        yield idx


def train(graphs: list[Graph], Y: np.ndarray, seed: int, cfg: TrainConfig = TrainConfig(),
          log=None) -> POM:
    """Train one model from scratch, deterministically on the CPU."""
    torch.set_num_threads(cfg.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = POM(n_tasks=Y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8)
    sched = torch.optim.lr_scheduler.StepLR(opt, cfg.decay_steps, cfg.decay_rate)
    w = imbalance_weights(Y)
    Yt = torch.from_numpy(np.asarray(Y, dtype=np.float32))
    model.train()
    for epoch in range(cfg.epochs):
        tot = 0.0
        for idx in _batches(len(graphs), cfg.batch_size, rng):
            b = collate([graphs[i] for i in idx])
            opt.zero_grad()
            logits, _ = model(b)
            loss = pom_loss(logits, Yt[idx], w) + regularization(model, cfg.weight_decay)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        if log:
            log(epoch + 1, tot)
    return model


@torch.no_grad()
def predict(model: POM, graphs: list[Graph], batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """(probabilities, 256-d embeddings) in eval mode."""
    model.eval()
    P, E = [], []
    for s in range(0, len(graphs), batch_size):
        logits, emb = model(collate(graphs[s:s + batch_size]))
        P.append(torch.sigmoid(logits).numpy())
        E.append(emb.numpy())
    return np.concatenate(P), np.concatenate(E)
