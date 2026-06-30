import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

MODEL_PATH = "model.joblib"


def main():
    # 1. Charger le dataset California Housing
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    # 2. Séparer les variables explicatives (X) et la variable cible (y)
    X = df[FEATURES]
    y = df["MedHouseVal"]

    # 3. Créer un jeu d'entraînement et un jeu de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Entraîner un modèle de régression (Random Forest)
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Évaluation rapide
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE  : {mse:.4f}")
    print(f"R2   : {r2:.4f}")

    # 5. Sauvegarder le modèle entraîné
    joblib.dump(model, MODEL_PATH)
    print(f"Modèle sauvegardé dans : {MODEL_PATH}")


if __name__ == "__main__":
    main()
