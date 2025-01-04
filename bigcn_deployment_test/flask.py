from flask import Flask, request, jsonify
import torch
from bigcn_deployment_test.main import load_model
from gnn import Model

app = Flask(__name__)

# Charger le modèle
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer l'article soumis via l'interface
    data = request.json
    article = data.get('article', '')
    
    if not article:
        return jsonify({'error': 'Veuillez fournir un article'}), 400
    
    # Prétraitement de l'article (ajouter votre méthode ici)
    # Exemple : transformer le texte en embeddings (à adapter selon votre modèle)
    processed_data = preprocess_article(article)
    
    # Convertir les données au format attendu par le modèle
    with torch.no_grad():
        prediction = model(processed_data)
        prob = torch.softmax(prediction, dim=1)
        fake_prob = prob[0][0].item()  # Probabilité "Fake"
        real_prob = prob[0][1].item()  # Probabilité "Réel"
    
    # Retourner le résultat
    return jsonify({
        'article': article,
        'fake_prob': fake_prob,
        'real_prob': real_prob,
        'prediction': 'Fake' if fake_prob > real_prob else 'Réel'
    })

def preprocess_article(article):
    # Placeholder pour transformer un article en format de graphe ou embeddings
    # À personnaliser selon votre modèle
    pass

if __name__ == '__main__':
    app.run(debug=True)
