import os
import sys
sys.path.append(os.getcwd())
import argparse
from tqdm import tqdm
import copy as cp

import torch
from torch.utils.data import random_split
from torch_scatter import scatter_mean
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel

from utils.data_loader import *
from utils.eval_helper import *

"""

The Bi-GCN is adopted from the original implementation from the paper authors 

Paper: Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks
Link: https://arxiv.org/pdf/2001.06362.pdf
Source Code: https://github.com/TianBian95/BiGCN

"""


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
import copy as cp

class TDrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = cp.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = cp.copy(x)
        
        # Ensure root_index exists
        if not hasattr(data, 'root_index'):
            # Assuming first node of each graph is root
            batch_size = data.batch.max().item() + 1
            root_indices = []
            for i in range(batch_size):
                batch_mask = data.batch == i
                graph_nodes = torch.nonzero(batch_mask).squeeze()
                root_indices.append(graph_nodes[0])
            data.root_index = torch.tensor(root_indices, device=x.device)

        rootindex = data.root_index
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(rootindex.device)
        batch_size = data.batch.max().item() + 1

        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = torch.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(rootindex.device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = torch.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)

        return x

class BUrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        # Ensure BU_edge_index exists
        if not hasattr(data, 'BU_edge_index'):
            data.BU_edge_index = data.edge_index.flip([0])
            
        x, edge_index = data.x, data.BU_edge_index
        x1 = cp.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = cp.copy(x)

        # Ensure root_index exists (same as TD)
        if not hasattr(data, 'root_index'):
            batch_size = data.batch.max().item() + 1
            root_indices = []
            for i in range(batch_size):
                batch_mask = data.batch == i
                graph_nodes = torch.nonzero(batch_mask).squeeze()
                root_indices.append(graph_nodes[0])
            data.root_index = torch.tensor(root_indices, device=x.device)

        rootindex = data.root_index
        root_extend = torch.zeros(len(data.batch), x1.size(1)).to(rootindex.device)
        batch_size = data.batch.max().item() + 1
        
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = torch.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(data.batch), x2.size(1)).to(rootindex.device)
        for num_batch in range(batch_size):
            index = (torch.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = torch.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)
        
        return x

class Net(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_classes=2):
        """
        Initialize BiGCN model
        
        Args:
            num_features (int): Number of input features
            hidden_size (int): Hidden layer size
            num_classes (int, optional): Number of output classes. Defaults to 2.
        """
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(num_features, hidden_size, hidden_size)
        self.BUrumorGCN = BUrumorGCN(num_features, hidden_size, hidden_size)
        self.fc = torch.nn.Linear((hidden_size + hidden_size) * 2, num_classes)

    def forward(self, data):
        """
        Forward pass through BiGCN
        
        Args:
            data: PyG Data object containing the graph
            
        Returns:
            torch.Tensor: Log probabilities for each class
        """
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = torch.cat((TD_x, BU_x), 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def compute_test(loader, verbose=False):
	model.eval()
	loss_test = 0.0
	out_log = []
	with torch.no_grad():
		for data in loader:
			if not args.multi_gpu:
				data = data.to(args.device)
			out = model(data)
			if args.multi_gpu:
				y = torch.cat([d.y for d in data]).to(out.device)
			else:
				y = data.y
			if verbose:
				print(F.softmax(out, dim=1).cpu().numpy())
			out_log.append([F.softmax(out, dim=1), y])
			loss_test += F.nll_loss(out, y).item()
	return eval_deep(out_log, loader), loss_test


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cpu', help='specify cuda devices')
# hyper-parameters
parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--TDdroprate', type=float, default=0.2, help='dropout ratio')
parser.add_argument('--BUdroprate', type=float, default=0.2, help='dropout ratio')
parser.add_argument('--epochs', type=int, default=45, help='maximum number of epochs')
parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
parser.add_argument('--feature', type=str, default='profile', help='feature type, [profile, spacy, bert, content]')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(args.seed)

dataset = FNNDataset(root='data/', feature=args.feature, empty=False, name=args.dataset,
					 transform=DropEdge(args.TDdroprate, args.BUdroprate))

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

print(args)

num_training = int(len(dataset) * 0.2)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

if args.multi_gpu:
	loader = DataListLoader
else:
	loader = DataLoader

train_loader = loader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = loader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = loader(test_set, batch_size=args.batch_size, shuffle=False)

model = Net(args.num_features, args.nhid, args.nhid)
if args.multi_gpu:
	model = DataParallel(model)
model = model.to(args.device)

if not args.multi_gpu:
	BU_params = list(map(id, model.BUrumorGCN.conv1.parameters()))
	BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
	base_params = filter(lambda p: id(p) not in BU_params, model.parameters())
	optimizer = torch.optim.Adam([
		{'params': base_params},
		{'params': model.BUrumorGCN.conv1.parameters(), 'lr': args.lr / 5},
		{'params': model.BUrumorGCN.conv2.parameters(), 'lr': args.lr / 5}
	], lr=args.lr, weight_decay=args.weight_decay)
else:
	BU_params = list(map(id, model.module.BUrumorGCN.conv1.parameters()))
	BU_params += list(map(id, model.module.BUrumorGCN.conv2.parameters()))
	base_params = filter(lambda p: id(p) not in BU_params, model.parameters())
	optimizer = torch.optim.Adam([
		{'params': base_params},
		{'params': model.module.BUrumorGCN.conv1.parameters(), 'lr': args.lr / 5},
		{'params': model.module.BUrumorGCN.conv2.parameters(), 'lr': args.lr / 5}
	], lr=args.lr, weight_decay=args.weight_decay)


if __name__ == "__main__":
    model.train()
    for epoch in tqdm(range(args.epochs)):
        out_log = []
        loss_train = 0.0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            if not args.multi_gpu:
                data = data.to(args.device)
            out = model(data)
            if args.multi_gpu:
                y = torch.cat([d.y for d in data]).to(out.device)
            else:
                y = data.y
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            out_log.append([F.softmax(out, dim=1), y])

        acc_train, _, _, _, recall_train, auc_train, _ = eval_deep(out_log, train_loader)
        [acc_val, _, _, _, recall_val, auc_val, _], loss_val = compute_test(val_loader)
        print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
              f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
              f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
              f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}')

    # Sauvegarder le modèle après l'entraînement
    torch.save(model.state_dict(), 'model.pth')
    torch.save(args, 'args.pth')
    print("Modèle sauvegardé dans 'model.pth' et les arguments dans 'args.pth'.")

    # Évaluation sur le test set
    [acc, f1_macro, f1_micro, precision, recall, auc, ap], test_loss = compute_test(test_loader, verbose=False)
    print(f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f},'
          f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}')

