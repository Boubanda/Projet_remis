import re
import unicodedata
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
import torch

# Téléchargement du tokenizer BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Chargement du modèle BERT pour l'analyse des sentiments
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Fonction pour nettoyer et préparer le texte pour BERT
def clean_text(text):
    """Nettoie le texte avant la tokenisation avec BERT."""
    # Normalisation Unicode (élimine les accents)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    
    # Suppression des chiffres et des caractères spéciaux, mais on conserve la structure du texte
    text = re.sub(r'\d+', '', text)  # Supprime les chiffres
    text = re.sub(r'[^\w\s]', '', text)  # Supprime les caractères spéciaux
    
    # Mise en minuscule
    text = text.lower().strip()
    
    return text

# Exemple de texte
sample_text = "Bonjour ! Ceci est un exemple avec des chiffres 123 et des accents éàô."

# Nettoyage du texte
cleaned_text = clean_text(sample_text)

# Tokenisation avec le tokenizer BERT
inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True)

# Prédiction des sentiments
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1)

# Affichage du résultat
print("Texte original :", sample_text)
print("Texte nettoyé :", cleaned_text)
print("Classe prédite pour l'analyse de sentiment : ", predicted_class.item())  # 0 ou 1 (selon l'entraînement du modèle)
