import argparse
import time
from tqdm import tqdm
import copy as cp
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel, GATConv
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool

from utils.data_loader import FNNDataset
from utils.eval_helper import eval_deep

class Net(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, concat=False):
        super(Net, self).__init__()
        self.num_features = in_feats  # Use in_feats from arguments
        self.num_classes = out_feats   # Use out_feats from arguments
        self.nhid = hid_feats           # Use hid_feats from arguments
        self.concat = concat

        self.conv1 = GATConv(self.num_features, self.nhid * 2)
        self.conv2 = GATConv(self.nhid * 2, self.nhid * 2)

        self.fc1 = Linear(self.nhid * 2, self.nhid)

        if self.concat:
            self.fc0 = Linear(self.num_features, self.nhid)
            self.fc1 = Linear(self.nhid * 2, self.nhid)

        self.fc2 = Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.selu(self.conv1(x, edge_index))
        x = F.selu(self.conv2(x, edge_index))
        x = F.selu(global_mean_pool(x, batch))
        x = F.selu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        if self.concat:
            news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
            news = F.relu(self.fc0(news))
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.fc1(x))

        x = F.log_softmax(self.fc2(x), dim=-1)

        return x

@torch.no_grad()
def compute_test(loader, model, args, verbose=False):
    model.eval()
    loss_test = 0.0
    out_log = []
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

def main():
    parser = argparse.ArgumentParser()

    # Original model parameters
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--device', type=str, default='cpu', help='specify cuda devices')

    # Hyper-parameters
    parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--nhid', type=int, default=128, help='hidden size')
    parser.add_argument('--epochs', type=int, default=60, help='maximum number of epochs')
    parser.add_argument('--concat', type=bool, default=False,
                        help='whether concat news embedding and graph embedding')
    parser.add_argument('--multi_gpu', type=bool,
                        default=False,
                        help='multi-gpu mode')
    parser.add_argument('--feature', type=str,
                        default='spacy',
                        help='feature type')

    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load dataset with all required parameters
    dataset = FNNDataset(
        root='data/',
        feature=args.feature,
        empty=False,
        name=args.dataset,
        transform=None
    )

    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features

    print(args)

    num_training = int(len(dataset) * 0.2)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)
    
    training_set, validation_set, test_set = random_split(dataset,
                                            [num_training, num_val, num_test])

    loader_class = DataListLoader if args.multi_gpu else DataLoader

    train_loader = loader_class(training_set,
                                batch_size=args.batch_size,
                                shuffle=True)
    
    val_loader = loader_class(validation_set,
                              batch_size=args.batch_size,
                              shuffle=False)
    
    test_loader = loader_class(test_set,
                               batch_size=args.batch_size,
                               shuffle=False)

    # Initialize model with appropriate parameters
    model = Net(args.num_features, args.nhid, args.num_classes, concat=args.concat).to(args.device)

    if args.multi_gpu:
        model = DataParallel(model)
    
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # Model training loop
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

        [acc_val, _, _, _, recall_val, auc_val, _], loss_val = compute_test(val_loader, model, args)

        print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
              f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
              f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
              f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}')

if __name__ == "__main__":
    main()