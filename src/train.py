from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def calculate_wcss(X, k_values):
    wcss_scores = []

    for k in k_values:
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=42)
        model.fit(X)
        wcss_scores.append(model.inertia_)

    return wcss_scores


def calculate_silhouette_scores(X, k_values):
    silhouette_scores = []

    for k in k_values:
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=42)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)

    return silhouette_scores


def find_best_k(k_values, silhouette_scores):
    best_index = silhouette_scores.index(max(silhouette_scores))
    best_k = k_values[best_index]
    return best_k


def train_kmeans(X, k):
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=42)
    model.fit(X)
    return model