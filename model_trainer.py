import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils.eval_helper import eval_deep

class ModelTrainer:
    def __init__(self, model_name, model, optimizer, device, train_loader, val_loader, test_loader):
        self.model_name = model_name
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Initialize metric tracking lists
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.train_aucs = []
        self.val_aucs = []
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        out_log = []
        
        for data in self.train_loader:
            self.optimizer.zero_grad()
            data = data.to(self.device)
            
            # Handle multi-GPU case
            if isinstance(data, list):
                out = self.model(data)
                y = torch.cat([d.y for d in data]).to(out.device)
            else:
                out = self.model(data)
                y = data.y
                
            loss = F.nll_loss(out, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            out_log.append([F.softmax(out, dim=1), y])
            
        # Calculate metrics using eval_deep
        acc, _, _, _, recall, auc, _ = eval_deep(out_log, self.train_loader)
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss, acc, auc
        
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        out_log = []
        
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                
                # Handle multi-GPU case
                if isinstance(data, list):
                    out = self.model(data)
                    y = torch.cat([d.y for d in data]).to(out.device)
                else:
                    out = self.model(data)
                    y = data.y
                    
                loss = F.nll_loss(out, y)
                total_loss += loss.item()
                out_log.append([F.softmax(out, dim=1), y])
                
        # Calculate all metrics
        acc, f1_macro, f1_micro, precision, recall, auc, ap = eval_deep(out_log, loader)
        avg_loss = total_loss / len(loader)
        
        return avg_loss, acc, auc, f1_macro, precision, recall
        
    def save_metrics(self, filename):
        """Save training metrics to a file"""
        metrics = {
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'train_aucs': self.train_aucs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'val_aucs': self.val_aucs
        }
        np.save(filename, metrics)
        
    def train(self, epochs):
        """Complete training loop with metric tracking"""
        print(f"Training {self.model_name}...")
        best_val_auc = 0
        
        for epoch in tqdm(range(epochs)):
            # Training phase
            train_loss, train_acc, train_auc = self.train_epoch()
            
            # Validation phase
            val_loss, val_acc, val_auc, val_f1, val_prec, val_rec = self.evaluate(self.val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.train_aucs.append(train_auc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            self.val_aucs.append(val_auc)
            
            if epoch % 5 == 0:
                print(f'Epoch {epoch}:')
                print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}')
                print(f'Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}')
                
            # Track best validation performance
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model = self.model.state_dict()
        
        # Restore best model
        self.model.load_state_dict(best_model)
        
        # Final test evaluation
        test_loss, test_acc, test_auc, test_f1, test_prec, test_rec = self.evaluate(self.test_loader)
        return test_loss, test_acc, test_auc, test_f1, test_prec, test_rec