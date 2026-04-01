import os
import logging
import warnings

from src.data_loader import load_data
from src.preprocessing import select_features, scale_features
from src.train import (
    calculate_wcss,
    calculate_silhouette_scores,
    find_best_k,
    train_kmeans
)
from src.save_model import save_pickle


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/app.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def create_folders():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)


def main():
    warnings.filterwarnings("ignore")
    os.environ["OMP_NUM_THREADS"] = "1"

    setup_logging()
    create_folders()

    logging.info("Program started.")

    df = load_data("data/mall_customers.csv")
    logging.info("Data loaded successfully.")

    X = select_features(df)
    logging.info("Features selected: Age, Annual_Income, Spending_Score.")

    X_scaled, scaler = scale_features(X)
    logging.info("Feature scaling completed.")

    k_values = list(range(3, 9))

    wcss_scores = calculate_wcss(X_scaled, k_values)
    logging.info(f"WCSS scores: {wcss_scores}")

    silhouette_scores = calculate_silhouette_scores(X_scaled, k_values)
    logging.info(f"Silhouette scores: {silhouette_scores}")

    best_k = find_best_k(k_values, silhouette_scores)
    logging.info(f"Best K selected: {best_k}")

    print("K values:", k_values)
    print("WCSS scores:", wcss_scores)
    print("Silhouette scores:", silhouette_scores)
    print("Best K:", best_k)

    model = train_kmeans(X_scaled, best_k)
    logging.info("KMeans model trained successfully.")

    save_pickle(model, "models/kmeans_model.pkl")
    save_pickle(scaler, "models/scaler.pkl")
    logging.info("Model and scaler saved successfully.")

    print("Model and scaler saved successfully.")


if __name__ == "__main__":
    main()