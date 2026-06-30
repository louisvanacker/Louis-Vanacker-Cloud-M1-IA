# Partie 7 - Réponses

1. **Pourquoi faut-il sauvegarder le modèle entraîné ?**
   Pour éviter de devoir le réentraîner à chaque fois que l'API démarre. L'entraînement
   peut être long et coûteux ; on l'exécute une fois, on sauvegarde le résultat
   (`model.joblib`), puis on le recharge instantanément à chaque démarrage.

2. **Pourquoi le modèle est-il chargé au démarrage de l'API ?**
   Pour ne le charger qu'une seule fois en mémoire et le réutiliser pour toutes les
   requêtes suivantes, au lieu de le recharger depuis le disque à chaque appel à
   `/predict`, ce qui serait beaucoup plus lent.

3. **Pourquoi utilise-t-on 0.0.0.0 dans Docker ?**
   Parce que `127.0.0.1` (localhost) à l'intérieur d'un conteneur ne désigne que le
   conteneur lui-même et n'est pas accessible depuis l'extérieur. `0.0.0.0` indique au
   serveur d'écouter sur toutes les interfaces réseau du conteneur, ce qui permet à
   Docker de rediriger le trafic depuis la machine hôte (via `-p 8000:8000`) vers l'API.

4. **Quelle est la différence entre entraîner un modèle et servir un modèle ?**
   Entraîner consiste à apprendre les paramètres du modèle à partir de données
   historiques (étape ponctuelle, généralement hors ligne, dans `train.py`). Servir
   consiste à exposer le modèle déjà entraîné pour qu'il réponde à des requêtes en
   temps réel (étape continue, en production, dans `app.py`).

5. **Pourquoi Docker est-il utile pour déployer une API ML ?**
   Docker encapsule le code, les dépendances (versions de Python, librairies ML) et la
   configuration dans une image reproductible. Cela garantit que l'API fonctionne de la
   même façon sur n'importe quelle machine (dev, test, prod), évite les problèmes de
   "ça marche chez moi" et facilite le déploiement/scalabilité sur des plateformes cloud.
