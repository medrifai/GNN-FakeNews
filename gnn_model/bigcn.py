import os
import sys
import traceback
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader

# Add necessary paths
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

from utils.data_loader import FNNDataset

class TDrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = x.float()
        x = self.conv1(x1, edge_index)
        x2 = x

        if not hasattr(data, 'root_index'):
            batch_size = 1 if data.batch is None else data.batch.max().item() + 1
            root_indices = []
            for i in range(batch_size):
                if data.batch is None:
                    batch_mask = torch.ones(x.size(0), dtype=torch.bool)
                else:
                    batch_mask = data.batch == i
                graph_nodes = torch.nonzero(batch_mask).squeeze()
                root_indices.append(graph_nodes[0])
            data.root_index = torch.tensor(root_indices, device=x.device)

        batch = data.batch if data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        root_extend = torch.zeros(len(batch), x1.size(1), device=x.device)
        batch_size = batch.max().item() + 1

        for num_batch in range(batch_size):
            index = (batch == num_batch)
            root_extend[index] = x1[data.root_index[num_batch]]

        x = torch.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        root_extend = torch.zeros(len(batch), x2.size(1), device=x.device)
        for num_batch in range(batch_size):
            index = (batch == num_batch)
            root_extend[index] = x2[data.root_index[num_batch]]

        x = torch.cat((x, root_extend), 1)
        x = scatter_mean(x, batch, dim=0)

        return x
class BUrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, data):
        if not hasattr(data, 'BU_edge_index'):
            data.BU_edge_index = data.edge_index.flip([0])

        x, edge_index = data.x, data.BU_edge_index
        x1 = x.float()
        x = self.conv1(x1, edge_index)
        x2 = x

        if not hasattr(data, 'root_index'):
            batch_size = 1 if data.batch is None else data.batch.max().item() + 1
            root_indices = []
            for i in range(batch_size):
                if data.batch is None:
                    batch_mask = torch.ones(x.size(0), dtype=torch.bool)
                else:
                    batch_mask = data.batch == i
                graph_nodes = torch.nonzero(batch_mask).squeeze()
                root_indices.append(graph_nodes[0])
            data.root_index = torch.tensor(root_indices, device=x.device)

        batch = data.batch if data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        root_extend = torch.zeros(len(batch), x1.size(1), device=x.device)
        batch_size = batch.max().item() + 1

        for num_batch in range(batch_size):
            index = (batch == num_batch)
            root_extend[index] = x1[data.root_index[num_batch]]

        x = torch.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        root_extend = torch.zeros(len(batch), x2.size(1), device=x.device)
        for num_batch in range(batch_size):
            index = (batch == num_batch)
            root_extend[index] = x2[data.root_index[num_batch]]

        x = torch.cat((x, root_extend), 1)
        x = scatter_mean(x, batch, dim=0)

        return x

# Nouvelle classe d'attention
class RumorAttention(torch.nn.Module):
    def __init__(self, in_features):
        super(RumorAttention, self).__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class Net(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_classes=2):
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(num_features, hidden_size, hidden_size)
        self.BUrumorGCN = BUrumorGCN(num_features, hidden_size, hidden_size)
        
        # Calculer la taille des features après concaténation
        concat_size = (hidden_size + hidden_size) * 2
        
        # Utiliser le module d'attention
        self.attention = RumorAttention(concat_size)
        self.fc = torch.nn.Linear(concat_size, num_classes)

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = torch.cat((TD_x, BU_x), 1)
        
        # Appliquer l'attention
        x = self.attention(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def train_epoch(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data in tqdm(train_loader, desc='Training'):
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        y = data.y.long()
            
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.max(1)[1]
        correct += pred.eq(y).sum().item()
        total += y.size(0)
        
    return total_loss / len(train_loader), correct / total

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            y = data.y.long()
            loss = F.nll_loss(out, data.y)
            
            total_loss += loss.item()
            pred = out.max(1)[1]
            correct += pred.eq(data.y).sum().item()
            total += y.size(0)
            
    return total_loss / len(loader), correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dataset', type=str, default='politifact')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--nhid', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--feature', type=str, default='profile')
    parser.add_argument('--min_samples', type=int, default=1)
    args = parser.parse_args()

    # Définir device
    device = torch.device(args.device)  # Créer l'instance de device

    # Définir current_dir avant son utilisation
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent / 'data'
    
    # Supprimer le fichier processed s'il existe
    processed_file = data_dir / args.dataset.lower() / 'processed' / f'{args.dataset}_{args.feature}_data.pt'
    if processed_file.exists():
        processed_file.unlink()
        print(f"Removed existing processed file: {processed_file}")
        
        # Supprimer aussi le dossier processed s'il existe
        processed_dir = processed_file.parent
        if processed_dir.exists():
            import shutil
            shutil.rmtree(processed_dir)
            print(f"Removed processed directory: {processed_dir}")
    
    print(f"Path in bigcn.py: {data_dir}")
    print(f"Data directory exists in bigcn.py: {data_dir.exists()}")
    
    # Setup data
    try:
        dataset = FNNDataset(
            root=str(data_dir),
            feature=args.feature,
            name=args.dataset,
        )
        
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        traceback.print_exc()
        return

    print(f"Dataset loaded successfully with {len(dataset)} samples")
    
    # Get dataset properties
    args.num_classes = dataset.num_classes if hasattr(dataset, 'num_classes') else 2
    args.num_features = dataset[0].x.size(1)  # Obtenir le nombre de features à partir du premier graphe
    
    print(f"Number of features: {args.num_features}")
    print(f"Number of classes: {args.num_classes}")
    
    # Calculate split sizes
    total_samples = len(dataset)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - val_size
    
    # Create data splits
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize model - CORRECTION ICI
    model = Net(args.num_features, args.nhid, args.num_classes).to(device)  # Utiliser l'instance device
    print(f"Model initialized and moved to {device}")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Ajouter le scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',           # On surveille la métrique à maximiser (accuracy)
        factor=0.5,          # Divise le LR par 2 quand on plateau
        patience=5,          # Attend 5 époques avant de réduire le LR
        verbose=True,        # Affiche les messages lors des changements de LR
        min_lr=1e-6         # LR minimum
    )
    
    # Training loop
    best_val_acc = 0
    best_model_path = Path('best_model.pth')
    patience = 10
    no_improve = 0

    args_dict = vars(args)
    args_path = Path('args.pth')
    torch.save(args_dict, args_path)
    print(f"Arguments saved to {args_path}")
    
    print("Starting training...")
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        # Ajuster le learning rate en fonction de la performance
        scheduler.step(val_acc)
        
        # Afficher le learning rate actuel
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch {epoch:02d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved with validation accuracy: {val_acc:.4f}')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Load best model and evaluate on test set
    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f'Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}')

if __name__ == "__main__":
    main()