import torch
import numpy as np
import scipy.sparse as sp
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from sklearn.metrics import roc_auc_score, accuracy_score
import networkx as nx
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from icecream import ic
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import PPI
from torch_geometric.utils import to_undirected, add_self_loops, to_dense_adj
from torch_geometric.transforms import NormalizeFeatures
from sklearn.model_selection import ParameterGrid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.tensorboard import SummaryWriter

print("Using device:", device)
ic.disable()
np.random.seed(114514)
torch.random.manual_seed(114514)


def evaluate_link_prediction(
    pos_edge_index, neg_edge_index, node_embeddings, edge_classifier
):
    with torch.no_grad():
        pos_edge_embeddings = get_edge_embeddings(pos_edge_index.T, node_embeddings)
        neg_edge_embeddings = get_edge_embeddings(neg_edge_index.T, node_embeddings)

        pos_scores = edge_classifier(pos_edge_embeddings)
        neg_scores = edge_classifier(neg_edge_embeddings)

        labels = torch.cat(
            [torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))]
        )
        scores = torch.cat([pos_scores, neg_scores]).squeeze()

        probabilities = torch.sigmoid(scores)

        return roc_auc_score(labels.cpu(), probabilities.cpu())


def split_edges(edges, num_nodes):
    all_edge_idx = np.random.permutation(np.arange(edges.shape[0]))
    num_test = max(int(np.floor(edges.shape[0] / 10.0)), 1)
    num_val = max(int(np.floor(edges.shape[0] / 20.0)), 1)
    test_edge_idx = all_edge_idx[:num_test]
    val_edge_idx = all_edge_idx[num_test : (num_test + num_val)]
    train_edge_idx = all_edge_idx[(num_test + num_val) :]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def is_member_numpy(a, adj_set):
        return (a[0], a[1]) in adj_set or (a[1], a[0]) in adj_set

    adj_list = train_edges
    negative_edges = []
    adj_set = set(map(tuple, adj_list))
    negative_edges = []
    with tqdm(
        total=len(train_edges), desc="Creating negative edges", unit="edge"
    ) as pbar:
        while len(negative_edges) < len(train_edges):
            size = (len(train_edges) - len(negative_edges)) * 2
            neg_edge_candidates = np.random.randint(0, num_nodes, size=(size, 2))
            filtered_edges = [
                tuple(edge)
                for edge in neg_edge_candidates
                if not is_member_numpy(edge, adj_set) and edge[0] != edge[1]
            ]
            negative_edges.extend(filtered_edges)
            negative_edges = negative_edges[: len(train_edges)]
            pbar.update(len(negative_edges) - pbar.n)

    negative_edges = np.array(negative_edges[: len(train_edges)])

    return train_edges, val_edges, test_edges, negative_edges


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def train_val_test_split(num_nodes, dataset):
    rng = np.random.default_rng(seed=42)
    all_indices = np.arange(num_nodes)
    rng.shuffle(all_indices)
    if dataset == "cora":
        train_end = int(num_nodes * 0.6)
        val_end = int(num_nodes * 0.8)
    elif dataset == "citeseer":
        train_end = int(num_nodes * 0.6)
        val_end = int(num_nodes * 0.8)
    elif dataset == "ppi":
        train_end = int(num_nodes * 0.7)
        val_end = int(num_nodes * 0.85)
    else:
        raise ValueError("unknown dataset!", dataset)

    idx_train = all_indices[:train_end]
    idx_val = all_indices[train_end:val_end]
    idx_test = all_indices[val_end:]

    return idx_train, idx_val, idx_test


def load_ppi_data(ppi_folder_path):
    feature_file = f"{ppi_folder_path}/ppi-feats.npy"
    features = np.load(feature_file)

    label_file = f"{ppi_folder_path}/ppi-class_map.json"
    with open(label_file, "r") as lf:
        labels = json.load(lf)

    graph_file = f"{ppi_folder_path}/ppi-G.json"
    with open(graph_file, "r") as gf:
        graph_json = json.load(gf)
        G = nx.DiGraph()
        for edge in graph_json["links"]:
            G.add_edge(edge["source"], edge["target"])

    id_map_file = f"{ppi_folder_path}/ppi-id_map.json"
    with open(id_map_file, "r") as imf:
        id_map = json.load(imf)

    num_classes = len(next(iter(labels.values())))
    label_matrix = np.zeros((len(id_map), num_classes))
    for node_id, class_list in labels.items():
        node_index = id_map[str(node_id)]
        label_matrix[node_index, :] = class_list

    walks_file = f"{ppi_folder_path}/ppi-walks.txt"
    with open(walks_file, "r") as wf:
        walks = wf.readlines()
        walks = [list(map(int, line.strip().split())) for line in walks]

    return G, features, label_matrix, id_map, walks


def load_data(path="./data/", dataset="cora", task="nc"):
    print(f"loading {dataset} dataset...")
    if dataset == "cora":
        idx_features_labels = np.genfromtxt(
            os.path.join(path, dataset, f"{dataset}.content"), dtype=np.dtype(str)
        )
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])

        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(
            os.path.join(path, dataset, f"{dataset}.cites"), dtype=np.int32
        )
        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
        ).reshape(edges_unordered.shape)
        edges = edges[~np.isnan(edges).any(axis=1)]  # 移除含有 NaN 的边
        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        idx_train, idx_val, idx_test = train_val_test_split(adj.shape[0], dataset)
        features = normalize(features)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense()))
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        train_edges, val_edges, test_edges, negative_edges = split_edges(
            edges, adj.shape[0]
        )
        assert not np.isnan(edges).any(), "edges contain NaN values"
    elif dataset == "citeseer":
        dataset = Planetoid(
            root="./data/citeseer", name="Citeseer", transform=NormalizeFeatures()
        )
        data = dataset[0]
        edge_index = to_undirected(data.edge_index, data.num_nodes)
        print("edge_index shape after to_undirected:", edge_index.shape)
        edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)
        print("edge_index shape after add_self_loops:", edge_index.shape)
        # train_edges, val_edges, test_edges, negative_edges = split_edges(edge_index.numpy(), data.num_nodes)
        adj = to_dense_adj(edge_index)[0]
        features = data.x
        labels = data.y
        ic(edge_index.shape)
        ic(data.train_mask.shape)
        train_edges = edge_index[:, data.train_mask.nonzero(as_tuple=True)[0]].t()
        val_edges = edge_index[:, data.val_mask.nonzero(as_tuple=True)[0]].t()
        test_edges = edge_index[:, data.test_mask.nonzero(as_tuple=True)[0]].t()
        ic(train_edges.shape)
        ic(val_edges.shape)
        ic(test_edges.shape)
        idx_train = torch.where(data.train_mask)[0]
        idx_val = torch.where(data.val_mask)[0]
        idx_test = torch.where(data.test_mask)[0]
        adj_set = set([tuple(x) for x in edge_index.t().numpy()])

        def is_member_numpy(a, adj_set):
            return (a[0], a[1]) in adj_set or (a[1], a[0]) in adj_set

        negative_edges = []
        with tqdm(
            total=len(train_edges), desc="Creating negative edges", unit="edge"
        ) as pbar:
            while len(negative_edges) < len(train_edges):
                size = (len(train_edges) - len(negative_edges)) * 2
                neg_edge_candidates = np.random.randint(0, adj.shape[0], size=(size, 2))
                filtered_edges = [
                    tuple(edge)
                    for edge in neg_edge_candidates
                    if not is_member_numpy(edge, adj_set) and edge[0] != edge[1]
                ]
                negative_edges.extend(filtered_edges)
                negative_edges = negative_edges[: len(train_edges)]
                pbar.update(len(negative_edges) - pbar.n)

        negative_edges = np.array(negative_edges[: len(train_edges)])
    elif dataset == "ppi":
        G, features, label_matrix, id_map, walks = load_ppi_data(
            os.path.join(path, "ppi")
        )
        adj = nx.adjacency_matrix(G)
        features = sp.csr_matrix(features, dtype=np.float32)
        labels = label_matrix
        edges = np.array(G.edges())
        edges = edges[~np.isnan(edges).any(axis=1)]
        idx_train, idx_val, idx_test = train_val_test_split(adj.shape[0], dataset)
        features = normalize(features)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense()))
        features = torch.FloatTensor(np.array(features.todense()))
        if task == "nc":
            labels = torch.LongTensor(labels)
        else:
            labels = torch.LongTensor(np.where(labels)[1])
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        train_edges, val_edges, test_edges, negative_edges = split_edges(
            edges, adj.shape[0]
        )
        assert not np.isnan(edges).any(), "edges contain NaN values"
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    return (
        adj,
        features,
        labels,
        idx_train,
        idx_val,
        idx_test,
        train_edges,
        val_edges,
        test_edges,
        negative_edges,
    )


# parse args
def parse_args():
    parser = argparse.ArgumentParser(description="Run GCN.")
    parser.add_argument("--dataset", nargs="?", default="cora", help="Dataset to use.")
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of epochs to train."
    )
    parser.add_argument(
        "task",
        nargs="?",
        default="nc",
        help="Task to perform. node_classification or link_prediction.",
    )
    return parser.parse_args()


args = parse_args()
writer = SummaryWriter(
    f'./runs/exp_{args.dataset}_{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}'
)

(
    adj,
    features,
    labels,
    idx_train,
    idx_val,
    idx_test,
    train_edges,
    val_edges,
    test_edges,
    negative_edges,
) = load_data(dataset=args.dataset)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class PairNorm(nn.Module):
    def __init__(self, scale=1.0, eps=1e-6):
        super(PairNorm, self).__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        x = x - mean
        std = x.norm(p=2, dim=1, keepdim=True)
        x = self.scale * x / (std + self.eps)
        return x


def drop_edge(adj, drop_rate):
    if not adj.is_sparse:
        adj = adj.to_sparse()
    adj_coo = adj.coalesce()

    indices = adj_coo.indices()
    values = adj_coo.values()

    n_edges = indices.shape[1]

    drop_num = int(n_edges * drop_rate)

    all_indices = torch.arange(n_edges)
    drop_indices = torch.randperm(n_edges)[:drop_num]

    keep_indices = torch.tensor(
        np.setdiff1d(all_indices.numpy(), drop_indices.numpy()), dtype=torch.long
    )

    indices = indices[:, keep_indices]
    values = values[keep_indices]

    adj_dropped = torch.sparse_coo_tensor(indices, values, adj_coo.size())

    adj_dropped = adj_dropped.to(adj.device)

    return adj_dropped


class GCN(nn.Module):
    def __init__(
        self,
        nfeat,
        nhid,
        nclass,
        dropout,
        num_layers=2,
        add_self_loops=True,
        drop_edge_rate=0.0,
        use_pairnorm=True,
        activation="relu",
    ):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.add_self_loops = add_self_loops
        self.drop_edge_rate = drop_edge_rate
        self.use_pairnorm = use_pairnorm
        self.activation = activation
        self.dropout = dropout

        # Define graph convolution layers
        self.gc_layers = nn.ModuleList()
        self.gc_layers.append(GraphConvolution(nfeat, nhid))
        for _ in range(num_layers - 2):
            self.gc_layers.append(GraphConvolution(nhid, nhid))
        self.gc_layers.append(GraphConvolution(nhid, nclass))

    def forward(self, x, adj):
        ic(type(adj))
        ic(adj.shape)
        if self.drop_edge_rate > 0:
            adj = drop_edge(adj, self.drop_edge_rate)

        if self.add_self_loops:
            adj = adj.to_dense() + torch.eye(adj.size(0), device=adj.device)

        for i in range(self.num_layers - 1):
            x = self.gc_layers[i](x, adj)
            if self.use_pairnorm:
                x = PairNorm()(x)
            x = self._apply_activation(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc_layers[-1](x, adj)

        return F.log_softmax(x, dim=1)

    def _apply_activation(self, x):
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "tanh":
            return torch.tanh(x)
        elif self.activation == "leaky_relu":
            return F.leaky_relu(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")


def add_self_loops(edge_index, num_nodes):
    loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index


ic(features.shape[1])
gcn_model = GCN(nfeat=features.shape[1], nhid=512, nclass=256, dropout=0.5)

gcn_model = gcn_model.to(device)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class NodeClassifier(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim1=32, hidden_dim2=16, dropout_rate=0.5
    ):
        super(NodeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)

        self.fc3 = nn.Linear(hidden_dim2, output_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, node_embeddings):
        x = F.relu(self.bn1(self.fc1(node_embeddings)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        logits = self.fc3(x)
        return logits


class NodeClassifier_PPI(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim1=32, hidden_dim2=16, dropout_rate=0.5
    ):
        super(NodeClassifier_PPI, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)

        self.fc3 = nn.Linear(hidden_dim2, output_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, node_embeddings):
        x = F.relu(self.bn1(self.fc1(node_embeddings)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        logits = self.fc3(x)
        return logits


class EdgeClassifier(nn.Module):
    def __init__(
        self,
        input_dim=14,
        hidden_dim1=32,
        hidden_dim2=16,
        output_dim=1,
        dropout_rate=0.5,
    ):
        super(EdgeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)

        self.fc3 = nn.Linear(hidden_dim2, output_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, edge_embeddings):
        x = F.relu(self.bn1(self.fc1(edge_embeddings)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = self.fc3(x)

        return torch.sigmoid(x)


def get_edge_embeddings(edge_index, node_embeddings):
    print("edge_index shape", edge_index.shape)
    max_index = node_embeddings.shape[0] - 1  # 最大的有效索引值
    if edge_index.max() > max_index or edge_index.min() < 0:
        raise ValueError(
            f"edge_index contains invalid indices. Valid index range: 0 to {max_index}, but found {edge_index.min()} to {edge_index.max()}"
        )
    return torch.cat(
        (node_embeddings[edge_index[0]], node_embeddings[edge_index[1]]), 1
    )


def get_node_embeddings(features, adj):
    gcn_model.eval()
    with torch.no_grad():
        x = F.relu(gcn_model.gc1(features, adj))
        return x.detach()


def compute_loss(pos_edge_index, neg_edge_index, node_embeddings, edge_classifier):
    pos_edge_embeddings = get_edge_embeddings(pos_edge_index.T, node_embeddings).to(
        device
    )
    neg_edge_embeddings = get_edge_embeddings(neg_edge_index.T, node_embeddings).to(
        device
    )
    pos_scores = edge_classifier(pos_edge_embeddings).squeeze()  # 添加 .squeeze()
    neg_scores = edge_classifier(neg_edge_embeddings).squeeze()  # 添加 .squeeze()
    labels_pos = torch.ones(pos_scores.size(0)).to(device)
    labels_neg = torch.zeros(neg_scores.size(0)).to(device)
    loss = F.binary_cross_entropy(
        torch.cat([pos_scores, neg_scores]), torch.cat([labels_pos, labels_neg])
    )

    return loss


def train_link_prediction(
    edge_classifier, optimizer, pos_edge_index, neg_edge_index, features, adj, epoch
):
    t = time.time()
    gcn_model.train()
    optimizer.zero_grad()

    node_embeddings = gcn_model(features, adj)

    loss = compute_loss(
        pos_edge_index, neg_edge_index, node_embeddings, edge_classifier
    )
    loss.backward()
    optimizer.step()
    # validation
    gcn_model.eval()
    node_embeddings = gcn_model(features, adj)
    # val_acc = evaluate_link_prediction(torch.LongTensor(val_edges), torch.LongTensor(negative_edges), node_embeddings, edge_classifier)
    val_loss = compute_loss(
        torch.LongTensor(val_edges),
        torch.LongTensor(negative_edges),
        node_embeddings,
        edge_classifier,
    )
    print(
        "训练阶段: {:04d}".format(epoch + 1),
        "损失: {:.4f}".format(loss.item()),
        "时间: {:.4f}s".format(time.time() - t),
    )
    # writer.add_scalar('Loss/train', loss.item(), epoch)
    # writer.add_scalar('Loss/validation', val_loss, epoch)
    return loss.item(), val_loss.item()


def compute_node_classification_loss(
    node_embeddings, node_classifier, labels, criterion=nn.CrossEntropyLoss()
):

    pred_labels = node_classifier(node_embeddings)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_labels = torch.sigmoid(pred_labels) > 0.5
    pred_labels_test = pred_labels[idx_test]
    ic(
        "pred_labels_test shape in func compute_node_classification_loss",
        pred_labels_test.shape,
    )
    ic("labels shape in func compute_node_classification_loss", labels.shape)
    loss = criterion(pred_labels_test, labels)

    return loss


def train_node_classification(
    model,
    node_classifier,
    features,
    adj,
    labels,
    idx_train,
    idx_val,
    optimizer,
    criterion,
    epochs,
):
    loss_values = []
    val_loss_values = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj).to(device)
        node_embeddings = model(features, adj)
        ic(node_embeddings.shape)
        pred = node_classifier(node_embeddings)
        loss = criterion(pred[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            node_embeddings = model(features, adj)
            pred = node_classifier(node_embeddings)
            loss_val = criterion(pred[idx_val], labels[idx_val])

        loss_values.append(loss.item())
        val_loss_values.append(loss_val.item())
        print(
            "Epoch: {:04d}".format(epoch + 1),
            "loss_train: {:.4f}".format(loss.item()),
            "loss_val: {:.4f}".format(loss_val.item()),
        )
        writer.add_scalar("Loss/train", loss.item(), epoch)
        writer.add_scalar("Loss/validation", loss_val, epoch)
    print("Training complete.")
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label="Training Loss")
    plt.plot(val_loss_values, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{args.dataset} Loss")
    plt.legend()
    plt.savefig(
        f'./log/{args.dataset}_loss_time_exp4__{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}.png'
    )


from sklearn.metrics import f1_score


def evaluate_node_classification(
    model,
    node_classifier,
    features,
    adj,
    labels,
    idx_test,
    dataset,
    batch_size=100,
    criterion=nn.CrossEntropyLoss(),
):
    model.eval()
    with torch.no_grad():
        node_embeddings = model(features, adj)
        loss_test = compute_node_classification_loss(
            node_embeddings, node_classifier, labels[idx_test], criterion
        )
        pred_labels = []
        for i in range(0, len(idx_test), batch_size):
            batch_idx = idx_test[i : i + batch_size]
            batch_pred_labels = node_classifier(node_embeddings[batch_idx])
            pred_labels.append(batch_pred_labels)
        pred_labels = torch.cat(pred_labels, dim=0)
        if dataset == "ppi":
            pred_labels = torch.sigmoid(pred_labels) > 0.5
            ic(labels[idx_test].cpu().numpy().shape)
            ic(pred_labels.cpu().numpy().shape)
            f1_test = f1_score(
                labels[idx_test].cpu().numpy(),
                pred_labels.cpu().numpy(),
                average="micro",
            )
            print(
                "Test set results:",
                "loss= {:.4f}".format(loss_test.item()),
                "micro F1 score= {:.4f}".format(f1_test),
            )
            return f1_test
        else:
            acc_test = accuracy(pred_labels, labels[idx_test])
            print(
                "Test set results:",
                "loss= {:.4f}".format(loss_test.item()),
                "accuracy= {:.4f}".format(acc_test),
            )
            return acc_test


nclass = labels.max().item() + 1


edge_classifier = EdgeClassifier(input_dim=512)
edge_classifier = edge_classifier.to(device)
if args.dataset == "ppi":
    node_classifier = NodeClassifier_PPI(input_dim=nclass, output_dim=nclass)
else:
    node_classifier = NodeClassifier(input_dim=nclass, output_dim=nclass)
node_classifier = node_classifier.to(device)
loss_values = []
val_loss_values = []


optimizer = optim.Adam(
    list(gcn_model.parameters()) + list(edge_classifier.parameters()),
    lr=0.001,
    weight_decay=5e-4,
)
if args.dataset == "ppi":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()


param_grid = [
    {
        "num_layers": [2, 3, 4],
        "add_self_loops": [True, False],
        "drop_edge_rate": [0.1, 0.0, 0.2],
        "use_pairnorm": [True, False],
        "activation": ["relu", "tanh", "leaky_relu"],
    }
]

best_acc = 0.0
best_hyperparameters = None
for params in ParameterGrid(param_grid):
    gcn_model = GCN(nfeat=features.shape[1], nhid=512, nclass=256, dropout=0.5)

    gcn_model = gcn_model.to(device)
    optimizer = optim.Adam(list(gcn_model.parameters()), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    print(f"Trying parameters: {params}")
    for epoch in range(args.epochs):
        t = time.time()
        loss, val_loss = train_link_prediction(
            edge_classifier,
            optimizer,
            torch.LongTensor(train_edges).to(device),
            torch.LongTensor(negative_edges).to(device),
            features.to(device),
            adj.to(device),
            epoch,
        )
        print("one epoch takes time: {:.4f}s".format(time.time() - t))
        loss_values.append(loss)
        val_loss_values.append(val_loss)
    gcn_model.eval()
    node_embeddings = gcn_model(features.to(device), adj.to(device))
    val_acc = evaluate_link_prediction(
        torch.LongTensor(val_edges),
        torch.LongTensor(negative_edges),
        node_embeddings,
        edge_classifier,
    )
    test_acc = evaluate_link_prediction(
        torch.LongTensor(test_edges),
        torch.LongTensor(negative_edges),
        node_embeddings,
        edge_classifier,
    )
    print("Validation ACC:", val_acc)
    print("Test ACC:", test_acc)
    with open(
        f"./log_report_lp/{args.dataset}_metrics_node_classification_exp4_.txt", "a"
    ) as f:
        f.write(f"Trying parameters: {params}\n")
        f.write(f"acc_test: {test_acc:.4f}\n")
        f.write(f"val_acc_values: {val_acc}\n")
        f.write(f"loss_values: {loss_values[-1]}\n")
        f.write(f"val_loss_values: {val_loss_values[-1]}\n")
        f.write(f"best_acc: {best_acc}\n")
        f.write(f"best_hyperparameters: {best_hyperparameters}\n")
        f.write(f"-------------------------------------------------\n")
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label="Training Loss")
    plt.plot(val_loss_values, label="Validation Loss")
    xlims = plt.gca().get_xlim()
    ylims = plt.gca().get_ylim()
    loss_text_position = (
        xlims[0] + 0.05 * (xlims[1] - xlims[0]),
        ylims[1] - 0.1 * (ylims[1] - ylims[0]),
    )
    acc_text_position = (
        xlims[0] + 0.55 * (xlims[1] - xlims[0]),
        ylims[1] - 0.1 * (ylims[1] - ylims[0]),
    )
    if args.dataset == "ppi":
        plt.text(*acc_text_position, f"Test Accuracy: {test_acc:.2f}", color="r")
    else:
        plt.text(*acc_text_position, f"Test Accuracy: {val_acc:.2f}", color="r")

    plt.xlabel("Epoch")
    plt.ylabel("Metrics")
    plt.title(f"{args.dataset} Metrics Node Classification")
    plt.legend()
    plt.savefig(
        f'./log_report_lp/{args.dataset}_metrics_time_exp4_{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}_test_acc_{test_acc:.2f}.png'
    )
    if val_acc > best_acc:
        best_acc = val_acc
        best_hyperparameters = params
