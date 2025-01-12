from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
from torch_geometric.data import Data
import os
import sys
from pathlib import Path
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import json
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de NLTK et téléchargement des ressources nécessaires
def setup_nltk():
    """Configure et télécharge les ressources NLTK nécessaires"""
    nltk_resources = ['punkt', 'stopwords', 'punkt_tab', 'averaged_perceptron_tagger']
    for resource in nltk_resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement de la ressource NLTK {resource}: {str(e)}")
            raise

# Initialisation de Flask
app = Flask(__name__)

# Configuration du chemin du projet
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# Import du modèle GNN
try:
    from gnn_model.bigcn import Net
except ImportError as e:
    logger.error(f"Erreur lors de l'importation du modèle GNN: {str(e)}")
    raise

def load_model():
    """Charge le modèle et ses arguments"""
    try:
        # Charger les arguments sauvegardés
        args_dict = torch.load('args.pth', map_location='cpu')
        
        # Initialiser le modèle
        model = Net(
            num_features=args_dict['num_features'],
            hidden_size=args_dict['nhid'],
            num_classes=args_dict['num_classes']
        ).to('cpu')
        
        # Charger les poids du modèle
        model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
        model.eval()
        
        logger.info("Modèle chargé avec succès")
        return model, args_dict
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        raise

def preprocess_text(text):
    """Prétraite le texte pour l'analyse"""
    try:
        # Convertir en minuscules
        text = text.lower()
        
        # Supprimer les URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Supprimer la ponctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenization avec spécification de la langue française
        tokens = word_tokenize(text, language='french')
        
        # Supprimer les stop words français
        stop_words = set(stopwords.words('french'))
        tokens = [token for token in tokens if token not in stop_words]
        
        return ' '.join(tokens)
    
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement du texte: {str(e)}")
        raise

def extract_features(text):
    """Extrait les features du texte en utilisant CamemBERT"""
    try:
        tokenizer = AutoTokenizer.from_pretrained('camembert-base')
        model = AutoModel.from_pretrained('camembert-base')
        
        # Prétraiter le texte
        processed_text = preprocess_text(text)
        
        # Tokenizer et obtenir les embeddings
        inputs = tokenizer(processed_text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Utiliser la moyenne des embeddings comme features
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings
    
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction des features: {str(e)}")
        raise

def create_single_node_graph(text, num_features):
    """Crée un graphe pour le texte d'entrée"""
    try:
        # Extraire les features du texte
        features = extract_features(text)
        
        # Adapter la dimension si nécessaire
        if features.shape[1] != num_features:
            if features.shape[1] < num_features:
                features = F.pad(features, (0, num_features - features.shape[1]))
            else:
                features = features[:, :num_features]
        
        # Créer l'edge_index pour un seul nœud
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # Créer l'objet Data
        data = Data(x=features, edge_index=edge_index)
        data.root_index = torch.tensor([0], device='cpu')
        
        return data
    
    except Exception as e:
        logger.error(f"Erreur lors de la création du graphe: {str(e)}")
        raise

# Configuration initiale
try:
    setup_nltk()
    model, args_dict = load_model()
except Exception as e:
    logger.error(f"Erreur lors de la configuration initiale: {str(e)}")
    model = None
    args_dict = None

@app.route('/')
def home():
    """Route pour la page d'accueil"""
    return render_template('fake_news.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Route pour la prédiction"""
    if model is None or args_dict is None:
        return jsonify({'error': 'Le modèle n\'est pas correctement initialisé'}), 500
        
    try:
        # Récupérer l'article
        data = request.json
        article = data.get('article', '')
        
        if not article:
            return jsonify({'error': 'Veuillez fournir un article'}), 400
        
        # Créer le graphe avec les features extraites
        processed_data = create_single_node_graph(article, args_dict['num_features'])
        
        # Faire la prédiction
        with torch.no_grad():
            prediction = model(processed_data)
            prob = torch.exp(prediction)  # Convertir log_softmax en probabilités
            fake_prob = float(prob[0][0])
            real_prob = float(prob[0][1])
        
        return jsonify({
            'fake_prob': round(fake_prob * 100, 2),
            'real_prob': round(real_prob * 100, 2),
            'prediction': 'Fake' if fake_prob > real_prob else 'Réel'
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        return jsonify({
            'error': 'Une erreur est survenue lors de l\'analyse',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    # Configuration du mode debug et du port
    debug_mode = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    app.run(
        debug=debug_mode,
        port=port,
        host='0.0.0.0'
    )