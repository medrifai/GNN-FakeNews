import argparse
import torch
from model_trainer import ModelTrainer
from utils.data_loader import FNNDataset
from torch_geometric.transforms import ToUndirected
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from utils.metrics_collector import MetricsCollector

from gnn_model.gnn import Model as GNNModel
from gnn_model.bigcn import Net as BiGCNModel
from gnn_model.gcnfn import Net as GCNFNModel

def train_model(model_name, model, train_loader, val_loader, test_loader, 
                device, epochs, lr, weight_decay):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    metrics_collector = MetricsCollector()
    
    trainer = ModelTrainer(
        model_name=model_name,
        model=model,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
    
    print(f"\nTraining the {model_name} model")
    
    for epoch in range(epochs):
        train_loss, train_acc, train_auc = trainer.train_epoch()
        val_loss, val_acc, val_auc, _, _, _ = trainer.evaluate(val_loader)
        
        # Add metrics for this epoch
        metrics_collector.add_metrics(
            train_loss=train_loss,
            train_acc=train_acc,
            train_auc=train_auc,
            val_loss=val_loss,
            val_acc=val_acc,
            val_auc=val_auc
        )
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch}: Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}')
            print(f'Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}')
    
    # Save the collected metrics
    metrics_collector.save_metrics(f'metrics_{model_name}.npy')
    
    # Get final test metrics
    test_metrics = trainer.evaluate(test_loader)
    return test_metrics, metrics_collector

def main():
    parser = argparse.ArgumentParser()
    
    # System parameters
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--device', type=str, default='cpu', help='device to use')
    parser.add_argument('--multi_gpu', action='store_true', help='use multiple GPUs if available')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--feature', type=str, required=True, help='feature type')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--epochs', type=int, default=45, help='number of epochs')
    
    # Model parameters
    parser.add_argument('--nhid', type=int, default=128, help='hidden size')
    parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
    parser.add_argument('--model', type=str, default='sage', choices=['gcn', 'sage', 'gat'], help='model type')
    parser.add_argument('--concat', action='store_true', help='whether to concat embeddings')
    parser.add_argument('--TDdroprate', type=float, default=0.2, help='TD drop rate')
    parser.add_argument('--BUdroprate', type=float, default=0.2, help='BU drop rate')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        args.device = 'cpu'
    device = torch.device(args.device)
    
    # Load dataset
    dataset = FNNDataset(root='data/', 
                        feature=args.feature,
                        empty=False,
                        name=args.dataset,
                        transform=None)

    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features
    
    print(args)
    
    # Split dataset
    num_training = int(len(dataset) * 0.2)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)
    
    training_set, validation_set, test_set = random_split(dataset,
                                            [num_training, num_val, num_test])

    # Create data loaders
    train_loader = DataLoader(training_set,
                            batch_size=args.batch_size,
                            shuffle=True)
    val_loader = DataLoader(validation_set,
                           batch_size=args.batch_size,
                           shuffle=False)
    test_loader = DataLoader(test_set,
                            batch_size=args.batch_size,
                            shuffle=False)

    # Define models with their configurations
    models = {
        'GNN': {
            'model': GNNModel(args),
            'epochs': 35,
            'lr': 0.01,
            'weight_decay': 0.01 
        },
        'BiGCN': {
            'model': BiGCNModel(args.num_features, args.nhid, args.num_classes),
            'epochs': 45,
            'lr': 0.01,
            'weight_decay': 0.001
        },
        'GCNFN': {
            'model': GCNFNModel(args.num_features, args.nhid, args.num_classes),
            'epochs': 60,
            'lr': 0.001,
            'weight_decay': 0.01 
        }
    }
    
    for model_name, config in models.items():
        model_instance = config['model'].to(device)
        test_metrics, _ = train_model(
            model=model_instance,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=config['epochs'],
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        print(f"\nFinal results for {model_name}:")
        print(f"Test Loss: {test_metrics[0]:.4f}")
        print(f"Test Accuracy: {test_metrics[1]:.4f}")
        print(f"Test AUC: {test_metrics[2]:.4f}")
        print(f"Test F1: {test_metrics[3]:.4f}")

if __name__ == "__main__":
    main()