# 1. Image Python légère
FROM python:3.11-slim

# 2. Répertoire de travail
WORKDIR /app

# 3. Copier les dépendances
COPY requirements.txt .

# 4. Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copier le code source (app.py, train.py, model.joblib)
COPY . .

# Entraîner le modèle au moment du build pour que model.joblib
# soit présent dans l'image (sinon il faut le copier depuis l'hôte
# avant le build : voir README)
RUN python train.py

# 6. Exposer le port 8000
EXPOSE 8000

# 7. Lancer l'application FastAPI avec Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
