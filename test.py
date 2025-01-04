import numpy as np
import matplotlib.pyplot as plt
from utils.metrics_collector import MetricsCollector

# Charger les métriques pour chaque modèle
gcnfn_metrics = MetricsCollector()
bigcn_metrics = MetricsCollector()
gnn_metrics = MetricsCollector()

try:
    # Charger les fichiers de métriques
    gcnfn_metrics.load_metrics('metrics_GCNFN.npy')
    bigcn_metrics.load_metrics('metrics_BiGCN.npy')
    gnn_metrics.load_metrics('metrics_GNN.npy')
except FileNotFoundError as e:
    print(f"Erreur lors du chargement des fichiers de métriques : {e}")
    exit(1)

# Vérifiez si les données nécessaires sont présentes
def contains_key(metrics_collector, key):
    return all(key in epoch_metrics for epoch_metrics in metrics_collector.val_metrics)

if not contains_key(gcnfn_metrics, 'accuracy') or \
   not contains_key(bigcn_metrics, 'accuracy') or \
   not contains_key(gnn_metrics, 'accuracy'):
    print("Les fichiers de métriques ne contiennent pas la clé 'accuracy' dans val_metrics.")
    exit(1)

# Comparer les courbes
plt.figure(figsize=(12, 6))

# Extraire les courbes d'accuracy
gcnfn_val_acc = [epoch['accuracy'] for epoch in gcnfn_metrics.val_metrics]
bigcn_val_acc = [epoch['accuracy'] for epoch in bigcn_metrics.val_metrics]
gnn_val_acc = [epoch['accuracy'] for epoch in gnn_metrics.val_metrics]

# Tracer les courbes
plt.plot(gcnfn_val_acc, label='GCNFN', marker='o')
plt.plot(bigcn_val_acc, label='BiGCN', marker='s')
plt.plot(gnn_val_acc, label='GNN', marker='^')

# Ajouter des titres et légendes
plt.title('Comparaison des accuracies de validation')
plt.xlabel('Épochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Afficher la figure
plt.show()
