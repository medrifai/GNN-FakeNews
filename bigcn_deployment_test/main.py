import torch

# Charger le modèle entraîné
from gnn import Model
model_path = 'model.pth'

def load_model():
    # Charger les arguments utilisés pour l'entraînement
    args = torch.load('args.pth')
    model = Model(args).to('cpu')  # Charger sur CPU pour le service
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Mode évaluation
    return model

model = load_model()
