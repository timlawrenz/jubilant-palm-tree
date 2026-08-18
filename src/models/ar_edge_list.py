"""Exp 3 — Autoregressive edge-list generator (Executive-Branch replacement).

Decoder-only transformer emitting the execution graph's edge list as a token
sequence conditioned on the motif array (Bill of Materials), replacing the
dense-matrix DiT denoiser. The rest of the pipeline (ConstraintSolver,
6-Laws validator, routing-fidelity harness) is unchanged.

Sequence layout (single vocabulary, positions carry roles):
  prefix : for i in range(num_nodes): [MOTIF_TOKEN(motif_i)]   (conditioning)
  body   : for edge in canonical order: [SRC, DST, TYPE, IDX]  then EOS
  pad    : PAD to batch max length (dynamic)

Token roles:
  0..127    node ids (src/dst)
  128 TYPE_EXEC, 129 TYPE_DATA
  130..133  IDX 0..3
  134..139  MOTIF token for motif id 1..6 (prefix only, never predicted)
  140 EOS | 141 PAD

Prefix positions are masked from the loss. Post-decode: parse quadruplets ->
[B,3,N,N] discrete adjacency via a [B,6,N,N] logit heatmap using the
INV_SIGMOID_ONE convention (+10/-10), so the EXISTING
ConstraintSolver.discretize_and_repair runs unchanged.
"""

import json
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ---- constants -------------------------------------------------------------
MAX_NODES = 128
MAX_SEQ = 2048          # hard cap; dataset max measured at 638
PAD_ID = 141
EOS_ID = 140
IDX_0 = 130             # idx class k maps to token IDX_0 + k
TYPE_EXEC = 128
TYPE_DATA = 129
MOTIF_BASE = 134        # motif id m (1..6) -> MOTIF_BASE + (m-1)
VOCAB_SIZE = 142

LOGIT_ON = 10.0
LOGIT_OFF = -10.0

MOTIF_IDS = {"Boundary": 1, "Sequence": 2, "Condition": 3, "Loop": 4,
             "State": 5, "Message": 6}
_VALID_MOTIF = (1, 2, 3, 4, 5, 6)


def motif_token(mid: int) -> int:
    return MOTIF_BASE + (mid - 1)


def canonical_edge_key(e: dict) -> tuple:
    return (e["source_node"], e["target_node"], e["edge_type"],
            min(int(e["input_index"]), 3))


def encode_graph(graph: dict) -> tuple[list[int], int]:
    """Return (token list, num_nodes) for a graph whose node ids are
    0..num_nodes-1 (permutation applied upstream)."""
    nodes = graph["nodes"]
    motif_map = {n["node_id"]: MOTIF_IDS.get(n["motif"], 0) for n in nodes}
    num_nodes = len(nodes)

    seq = []
    for i in range(num_nodes):
        mid = motif_map.get(i, 0)
        if mid not in _VALID_MOTIF:
            mid = 1
        seq.append(motif_token(mid))

    for e in sorted(graph["edges"], key=canonical_edge_key):
        s = min(max(int(e["source_node"]), 0), MAX_NODES - 1)
        d = min(max(int(e["target_node"]), 0), MAX_NODES - 1)
        t = TYPE_EXEC if int(e["edge_type"]) == 0 else TYPE_DATA
        idx = IDX_0 + min(int(e["input_index"]), 3)
        seq.extend([s, d, t, idx])
    seq.append(EOS_ID)
    return seq, num_nodes


def decode_edge_tokens(tokens: list[int], num_nodes: int) -> list[tuple]:
    """Parse quadruplet body tokens after the prefix into edge tuples
    (src, dst, edge_type, input_index). Stops at EOS/PAD or first malformed
    quadruplet."""
    body = tokens[num_nodes:]
    edges = []
    i = 0
    while i + 3 < len(body):
        s, d, t, idx = body[i], body[i + 1], body[i + 2], body[i + 3]
        if s in (PAD_ID, EOS_ID) or d in (PAD_ID, EOS_ID):
            break
        if t != TYPE_EXEC and t != TYPE_DATA:
            break
        if not (IDX_0 <= idx <= IDX_0 + 3):
            break
        edges.append((s, d, 1 if t == TYPE_DATA else 0, idx - IDX_0))
        i += 4
    return edges


def logit_heatmap_from_edges(edges: list[tuple], num_nodes: int,
                             N: int = MAX_NODES) -> torch.Tensor:
    """[6, N, N] float logits (+10/-10) matching the DiT's continuous output
    channels (presence, edge_type, 4 index classes) so the ConstraintSolver
    consumes it verbatim. Self-loops and out-of-block edges dropped."""
    x = torch.full((6, N, N), LOGIT_OFF, dtype=torch.float32)
    for (s, d, etype, idx) in edges:
        if 0 <= s < num_nodes and 0 <= d < num_nodes and s != d:
            x[0, s, d] = LOGIT_ON
            x[1, s, d] = LOGIT_ON if etype == 1 else LOGIT_OFF
            if etype == 1:
                x[2 + idx, s, d] = LOGIT_ON
    return x


# ---- dataset (held-out split guardrail) ------------------------------------

class ARGraphDataset(Dataset):
    """Same 3,951-graph corpus as prior arms (max_nodes<=128 filter) with the
    pre-registered held-out split: eval graphs are disjoint from training.

    split_seed + holdout_frac split the corpus deterministically. eval_only
    arm returns only held-out graphs (capped at max_graphs for the N=512 eval);
    train arm returns the complement.
    """

    def __init__(self, jsonl_path, max_nodes=MAX_NODES, augment_permutation=True,
                 split_seed=42, holdout_frac=0.16, eval_only=False,
                 max_graphs=512):
        self.max_nodes = max_nodes
        self.augment_permutation = augment_permutation
        self.split_seed = split_seed
        self.eval_only = eval_only

        with open(jsonl_path) as f:
            all_graphs = []
            for line in f:
                if not line.strip():
                    continue
                g = json.loads(line)["compressed_graph"]
                if len(g["nodes"]) <= self.max_nodes:
                    all_graphs.append(g)

        n = len(all_graphs)
        rng = random.Random(split_seed)
        order = list(range(n))
        rng.shuffle(order)
        n_hold = max(1, int(round(n * holdout_frac)))
        hold = set(order[:n_hold])

        if eval_only:
            idxs = sorted(hold)
            if len(idxs) > max_graphs:
                idxs = idxs[:max_graphs]
        else:
            idxs = sorted(set(order) - hold)
        self.graphs = [all_graphs[i] for i in idxs]
        self.hold = hold
        print(f"[ARDataset] {'EVAL' if eval_only else 'TRAIN'}: {len(self.graphs)} graphs "
              f"(holdout={len(hold)})", flush=True)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        if self.augment_permutation:
            n = len(g["nodes"])
            perm = torch.randperm(n).tolist()
            old_ids = [x["node_id"] for x in g["nodes"]]
            mp = {old_ids[i]: perm[i] for i in range(n)}
            g = {
                "nodes": [{"node_id": mp[x["node_id"]], "motif": x["motif"]}
                          for x in g["nodes"]],
                "edges": [{"source_node": mp[e["source_node"]],
                           "target_node": mp[e["target_node"]],
                           "edge_type": e["edge_type"],
                           "input_index": e["input_index"]}
                          for e in g["edges"]],
            }
        seq, num_nodes = encode_graph(g)
        if len(seq) > MAX_SEQ:
            seq = seq[:MAX_SEQ - 1] + [EOS_ID]
        return torch.tensor(seq, dtype=torch.long), num_nodes


def collate_ar(batch):
    seqs, num_nodes = zip(*batch)
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), PAD_ID, dtype=torch.long)
    valid = torch.zeros((len(seqs), L), dtype=torch.bool)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = s
        valid[i, :len(s)] = True
    return x, valid, torch.tensor(num_nodes, dtype=torch.long)


# ---- decoder-only model ----------------------------------------------------

class EdgeListDecoder(nn.Module):
    """GPT-style decoder. Condition = prefix tokens (per-node motif class);
    output = body edge tokens. Causal self-attention, learned positions."""

    def __init__(self, d_model=256, n_heads=8, n_layers=6, vocab=VOCAB_SIZE,
                 max_len=MAX_SEQ, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                       dim_feedforward=1024, dropout=dropout,
                                       batch_first=True, activation="gelu",
                                       norm_first=True)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)
        self.max_len = max_len

    def _attn_mask(self, S, device):
        # causal: upper-triangular True = masked out (for F.multi_head / etc.)
        return torch.triu(torch.ones(S, S, dtype=torch.bool, device=device), 1)

    def forward(self, x, valid):
        """x: [B, S] token ids; valid: [B, S] True for real tokens.
        Returns logits [B, S, vocab] for every position.

        POSITION RELATIVE TO BODY (Exp 3 fix): the prefix length varies per
        graph (num_nodes), so absolute positions would shift the edge grammar's
        coordinates between samples and the model collapses to EOS-heavy
        degeneracy. We re-base positions so the BODY always starts at position
        MAX_NODES: prefix keeps 0..p-1, body token at sequence index i (i >= p)
        gets position MAX_NODES + (i - p). The first edge token therefore
        always sits on the same positional coordinate across samples."""
        B, S = x.shape
        arange = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
        # Prefix = LEADING RUN of motif tokens (134..139) ONLY. Naive
        # `x >= MOTIF_BASE` also counts EOS(140)/PAD(141) and would treat the
        # whole padded sequence as prefix, silently reverting to absolute
        # positions (the collapse bug). Count the run, then stop at the first
        # non-motif token.
        is_motif = (x >= MOTIF_BASE) & (x < MOTIF_BASE + 6)   # [B, S]
        # leading-run length = first index where is_motif is False
        not_motif = ~is_motif
        prefix_len = not_motif.long().argmax(dim=1)  # [B] (0 if all motifs — impossible)
        # guard: if a row is ALL motif tokens (degenerate), treat as 0 prefix
        all_motif = is_motif.all(dim=1)
        prefix_len = torch.where(all_motif, torch.zeros_like(prefix_len), prefix_len)
        prefix_len = prefix_len.unsqueeze(1)                  # [B, 1]
        body_mask = arange >= prefix_len
        pos_ids = torch.where(body_mask,
                              MAX_NODES + (arange - prefix_len),
                              arange)
        pos_ids = pos_ids.clamp(max=self.max_len - 1)
        h = self.tok(x) + self.pos(pos_ids)
        attn_mask = self._attn_mask(S, x.device)
        key_padding = ~valid
        for blk in self.blocks:
            h = blk(h, src_key_padding_mask=key_padding, src_mask=attn_mask)
        return self.head(self.final_norm(h))

    @torch.no_grad()
    def sample_greedy(self, prefix: torch.Tensor, num_nodes: int,
                      max_new: int = 2048) -> list[int]:
        """Emit edge tokens autoregressively starting from a prefix [S0] tensor.
        Returns the full token list (prefix + body)."""
        self.eval()
        device = next(self.parameters()).device
        seq = prefix.clone().to(device)
        for _ in range(max_new):
            L = seq.shape[0]
            if L >= MAX_SEQ:
                break
            valid = torch.ones(1, L, dtype=torch.bool, device=device)
            logits = self.forward(seq.unsqueeze(0), valid)[0, -1]
            nxt = int(logits.argmax().item())
            if nxt == EOS_ID:
                seq = torch.cat([seq, torch.tensor([EOS_ID], device=device)])
                break
            seq = torch.cat([seq, torch.tensor([nxt], device=device)])
        return seq.cpu().tolist()


def build_prefix(num_nodes: int, motifs: torch.Tensor, device=None) -> torch.Tensor:
    """Build the conditioning prefix from a [N] motif id tensor (0 for pad)."""
    mids = motifs[:num_nodes]
    toks = []
    for mid in mids.tolist():
        if mid not in _VALID_MOTIF:
            mid = 1
        toks.append(motif_token(mid))
    return torch.tensor(toks, dtype=torch.long, device=device)