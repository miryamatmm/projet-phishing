# === Imports ===
import os
import re
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import DBSCAN
from IPython.display import display
from sklearn.manifold import trustworthiness
import umap
import hdbscan
import subprocess
import json
import ast
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")


# ================================ Fonctions utilitaires ================================

def get_palette():
    """Retourne la palette de couleurs utilisée pour les e-mails."""
    return {
        "Safe Email": "#a7c7e7",   # bleu clair
        "Phishing Email": "#e75480"  # rose vif
    }


def preprocess_features(X, y=None):
    """Normalise X et encode y si fourni."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if y is not None:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return X_scaled, y_encoded, le
    return X_scaled

# ================================ Exploration descriptive ================================

def plot_text_length_distribution(
    df,
    text_col="text_length",
    label_col="Email Type",
    title=None,
    bins=60,
    cleaned=False,
    palette=None
):
    """Affiche la distribution des longueurs de texte selon le type d'e-mail."""
    sns.set_style("whitegrid")
    if palette is None:
        palette = get_palette()

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x=text_col,
        hue=label_col,
        bins=bins,
        alpha=0.85,
        multiple="dodge",
        palette=palette
    )

    if title is None:
        title = (
            "Distribution des longueurs d’e-mails après nettoyage"
            if cleaned else
            "Distribution initiale des longueurs d’e-mails"
        )

    plt.title(title, fontsize=13, color="#cc0066")
    plt.xlabel("Longueur du texte" + (" nettoyé" if cleaned else ""))
    plt.ylabel("Nombre d’e-mails")
    plt.tight_layout()
    plt.show()


def extract_nlp_features(text):
    """Extrait des caractéristiques linguistiques et structurelles d’un texte."""
    text_lower = text.lower()
    words = re.findall(r"\b\w+\b", text_lower)

    return {
        "text_length": len(text),
        "word_count": len(words),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        "nb_exclamations": text.count("!"),
        "nb_questions": text.count("?"),
        "nb_dollar": text.count("$"),
        "nb_percent": text.count("%"),
        "nb_points": text.count("."),
        "nb_commas": text.count(","),
        "nb_uppercase": sum(1 for c in text if c.isupper()),
        "ratio_uppercase": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "nb_digits": sum(c.isdigit() for c in text),
        "ratio_digits": sum(c.isdigit() for c in text) / max(len(text), 1),
        "contains_html": int(bool(re.search(r"<(html|div|a|table|img|span|br)", text_lower))),
        "nb_entities": len(re.findall(r"&[a-z]+;", text_lower))
    }


def analyze_nlp_features(df, label_col="Email Type", diff_threshold=0.5,
                         corr_threshold=0.8, top_n=10):
    """Analyse exploratoire des features NLP quantitatives."""
    print("Analyse des features NLP numériques...")

    numeric_cols = df.select_dtypes(include="number").columns
    means = df.groupby(label_col)[numeric_cols].mean().T
    means.columns = means.columns.str.strip()
    means["diff_abs"] = abs(means.get("Phishing Email", 0) - means.get("Safe Email", 0))
    means = means.sort_values(by="diff_abs", ascending=False)

    display(
        means.style
        .set_caption("Moyennes par type et différences absolues")
        .set_properties(**{"text-align": "center"})
        .background_gradient(cmap="RdPu", subset=["diff_abs"])
    )

    # Top features les plus discriminantes (> diff_threshold)
    top_features = means.head(top_n)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_features.index, y=top_features["diff_abs"], palette="RdPu")
    plt.title("Différence moyenne entre Phishing et Safe Emails", color="#e75480", fontsize=14)
    plt.ylabel("|Différence absolue|")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Corrélation entre features (pas avoir 2 fois la même info)
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[num_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap="RdPu", center=0, linewidths=0.5)
    plt.title("Corrélation entre les features NLP", color="#e75480")
    plt.tight_layout()
    plt.show()

    # Sélection finale des features
    selected = means[means["diff_abs"] > diff_threshold].index.tolist()
    final_features = []
    for f in selected:
        if all(abs(corr_matrix[f][final_features]) < corr_threshold):
            final_features.append(f)

    print(f"{len(final_features)} features finales sélectionnées :")
    print(final_features)

    return {
        "means": means,
        "corr_matrix": corr_matrix,
        "final_features": final_features
    }

# ================================ Encodage des e-mails (Embeddings) ================================

def get_or_build_embeddings(
    df_texts,
    text_col="Email Text",
    model_name="all-MiniLM-L6-v2",
    save_path="data/final_embeddings.parquet"
):
    """
    Retourne le DataFrame d'embeddings pour un jeu d'e-mails.
    Si le fichier existe déjà, le charge directement.
    Sinon, encode les textes avec SentenceTransformer, sauvegarde, puis le retourne.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        print(f"Chargement des embeddings existants depuis {save_path}...")
        return pd.read_parquet(save_path)

    print(f"Encodage des e-mails avec {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df_texts[text_col].tolist(), show_progress_bar=True)

    embeddings_df = pd.DataFrame(embeddings, columns=[f"embed_{i}" for i in range(embeddings.shape[1])])
    final_df = pd.concat([df_texts.reset_index(drop=True), embeddings_df], axis=1)
    final_df.to_parquet(save_path, index=False)

    print(f"Embeddings sauvegardés dans {save_path}")
    return final_df


# ================================ Réduction de dimension ================================

# ------------- PCA 2D -------------

def run_pca_2d(X_scaled, y, y_encoded, palette):
    """Exécute une PCA 2D, affiche la visualisation et retourne (X_pca2, score)."""
    pca = PCA(n_components=2, random_state=42)
    X_pca2 = pca.fit_transform(X_scaled)
    score_pca2 = silhouette_score(X_pca2, y_encoded, metric="euclidean")

    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        x=X_pca2[:, 0], y=X_pca2[:, 1],
        hue=y, palette=palette, s=18, alpha=0.8
    )
    plt.title(f"PCA 2D — silhouette = {score_pca2:.4f}")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(title="Email Type")
    plt.tight_layout()
    plt.show()

    return X_pca2, score_pca2

# ------------- t-SNE -------------

def run_umap_final(
    X,
    y=None,
    n_neighbors=30,
    min_dist=0.1,
    metric="cosine",
    n_components=2,
    n_epochs=500,
    densmap=True,
    cache_path="data/umap_final.npy",
    random_state=42,
    palette=None,
    plot=True
):
    """
    UMAP final — version stable et douce
    Réduction de dimension directe sur les embeddings (sans PCA, sans normalisation).
    Optimisée pour visualiser et identifier les types de phishing.
    """
    
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if palette is None:
        palette = get_palette()

    # Chargement du cache si existant
    if os.path.exists(cache_path):
        print(f"UMAP déjà calculé → Chargement depuis {cache_path}")
        X_umap = np.load(cache_path)
    else:
        print("Calcul en cours de la réduction UMAP")

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            n_components=n_components,
            n_epochs=n_epochs,
            densmap=densmap,
            random_state=random_state,
            low_memory=True
        )

        X_umap = reducer.fit_transform(X)
        np.save(cache_path, X_umap)
        print(f"UMAP sauvegardé dans {cache_path}")


    # Visualisation
    if plot and n_components == 2:
        plt.figure(figsize=(7, 6))
        if y is not None:
            sns.scatterplot(
                x=X_umap[:, 0],
                y=X_umap[:, 1],
                hue=y,
                s=15,
                alpha=0.75,
                palette=palette,
                edgecolor="white",
                linewidth=0.2
            )

        else:
            plt.scatter(X_umap[:, 0], X_umap[:, 1], s=8, alpha=0.7, color="#e6a4b4")

        plt.title(
            f"UMAP final — n_neighbors={n_neighbors}, min_dist={min_dist}",
            color="#e75480",
            fontsize=12
        )
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.tight_layout()
        plt.show()

    tw = trustworthiness(X, X_umap, n_neighbors=min(30, n_neighbors))
    print(f"Trustworthiness : {tw:.3f}")

    return X_umap


# ------------- UMAP -------------

def run_tsne_final(
    X,
    y=None,
    perplexity=30,
    learning_rate=200,
    max_iter=1000,
    n_components=2,
    metric="cosine",
    cache_path="data/tsne_final.npy",
    random_state=42,
    palette=None,
    plot=True
):
    """
    t-SNE final — version stable et douce
    Réduction directe sur les embeddings (sans PCA ni normalisation).
    Style visuel identique à run_umap_final.
    """

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if palette is None:
        palette = get_palette()

    # Chargement du cache
    if os.path.exists(cache_path):
        print(f"t-SNE déjà calculé → Chargement depuis {cache_path}")
        X_tsne = np.load(cache_path)
    else:
        print("Calcul t-SNE en cours…")

        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            metric=metric,
            init="pca",
            max_iter=max_iter,
            random_state=random_state,
            verbose=1,
        )

        X_tsne = tsne.fit_transform(X)
        np.save(cache_path, X_tsne)
        print(f"t-SNE sauvegardé dans {cache_path}")

    # Visualisation
    if plot and n_components == 2:
        plt.figure(figsize=(7, 6))
        if y is not None:
            sns.scatterplot(
                x=X_tsne[:, 0],
                y=X_tsne[:, 1],
                hue=y,
                s=15,
                alpha=0.75,
                palette=palette,
                edgecolor="white",
                linewidth=0.2
            )
        else:
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=8, alpha=0.7, color="#e6a4b4")

        plt.title(
            f"t-SNE final — perplexity={perplexity}, lr={learning_rate}",
            color="#e75480",
            fontsize=12
        )
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.show()

    tw = trustworthiness(X, X_tsne, n_neighbors=min(30, perplexity))
    print(f"Trustworthiness : {tw:.3f}")

    return X_tsne

# ================================ Clustering ================================

def explore_dbscan(X_emb, eps_values=None, min_samples_values=None, min_clusters_range=(5, 50)):
    """
    Exploration étendue de DBSCAN — sans normalisation ni silhouette.
    Objectif : identifier les zones de paramètres produisant des clusters exploitables.
    """

    print("Exploration étendue DBSCAN...")

    # grilles par défaut
    if eps_values is None:
        eps_values = [0.1, 0.3, 0.5, 0.75, 0.8, 0.9, 1, 1.5, 2, 2.5, 3, 4, 5, 10]
    if min_samples_values is None:
        min_samples_values = [2, 3, 5, 7, 10, 15, 20]

    results = []

    # boucle d'exploration
    for eps in eps_values:
        for ms in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X_emb)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            bruit_pct = (n_noise / len(labels)) * 100
            results.append((eps, ms, n_clusters, n_noise, bruit_pct))

    df_results = pd.DataFrame(results, columns=["eps", "min_samples", "clusters", "bruit", "bruit_%"])

    # on filtre selon une plage
    low, high = min_clusters_range
    df_valid = df_results[(df_results["clusters"] >= low) & (df_results["clusters"] <= high)]

    if df_valid.empty:
        print(f"Aucune configuration dans la plage [{low}, {high}] clusters.")
        return df_results, None, None

    # ratio = (nombre de clusters / (bruit% + 1)) -> favorise bcp de clusters et peu de bruit
    df_valid["score_ratio"] = df_valid["clusters"] / (df_valid["bruit_%"] + 1)

    df_valid = df_valid.sort_values(by="score_ratio", ascending=False).reset_index(drop=True)

    display(
        df_valid.head(10).style
        .background_gradient(cmap="RdPu", subset=["clusters"])
        .background_gradient(cmap="Greys", subset=["bruit_%"])
        .background_gradient(cmap="PuRd", subset=["score_ratio"])
        .set_caption(f"DBSCAN — Top 10 configurations [{low}, {high}] clusters (par score ratio)")
        .set_properties(**{"text-align": "center"})
    )

    # Sélection du meilleur compromis
    best = df_valid.iloc[0]
    best_eps = float(best["eps"])
    best_ms = int(best["min_samples"])

    # Clustering final
    db = DBSCAN(eps=best_eps, min_samples=best_ms)
    labels_best = db.fit_predict(X_emb)
    n_clusters_best = len(set(labels_best)) - (1 if -1 in labels_best else 0)
    n_noise_best = np.sum(labels_best == -1)
    bruit_pct_best = (n_noise_best / len(labels_best)) * 100

    # Visualisation
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        x=X_emb[:, 0], y=X_emb[:, 1],
        hue=labels_best, palette="Spectral", s=20, alpha=0.85, legend=False
    )
    plt.title(
        f"DBSCAN final — {n_clusters_best} clusters, {bruit_pct_best:.1f}% de bruit\n"
        f"(eps={best_eps}, min_samples={best_ms})",
        color="#e75480"
    )
    plt.tight_layout()
    plt.show()

    print(f"Meilleur résultat : {n_clusters_best} clusters — {bruit_pct_best:.1f}% de bruit")
    print(f"Paramètres : eps={best_eps}, min_samples={best_ms}")

    return df_valid, best, labels_best

def explore_hdbscan(
    X_emb,
    min_cluster_sizes=None,
    min_samples_values=None,
    target_min=10,
    target_max=50
):
    """
    Explore plusieurs configurations HDBSCAN et sélectionne celles donnant un nombre
    de clusters compris entre [target_min, target_max].
    Retourne (df_results, best, labels_best).
    """

    print("Exploration ciblée HDBSCAN...")

    if min_cluster_sizes is None:
        min_cluster_sizes = [30, 50, 80, 100, 150, 200]
    if min_samples_values is None:
        min_samples_values = [5, 10, 15, 20]
        

    results = []
    for mcs in min_cluster_sizes:
        for ms in min_samples_values:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=int(mcs),
                min_samples=int(ms),
                metric="euclidean"
            )
            labels = clusterer.fit_predict(X_emb)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            pct_noise = n_noise / len(labels) * 100
            results.append((int(mcs), int(ms), n_clusters, n_noise, pct_noise))

    df_results = pd.DataFrame(
        results,
        columns=["min_cluster_size", "min_samples", "clusters", "bruit", "bruit_%"]
    )

    # Filtrer pour garder les résultats entre 10 et 30 clusters
    df_filtered = df_results[
        (df_results["clusters"] >= target_min) &
        (df_results["clusters"] <= target_max)
    ].sort_values(by=["clusters", "bruit_%"], ascending=[False, True])

    if df_filtered.empty:
        print(f"Aucun résultat avec {target_min} ≤ clusters ≤ {target_max}.")
        print("→ Conseil : essayez des valeurs plus petites pour min_cluster_size.")
        return df_results, None, None

    best = df_filtered.iloc[0]
    best_mcs, best_ms = int(best["min_cluster_size"]), int(best["min_samples"])

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=best_mcs,
        min_samples=best_ms,
        metric="euclidean"
    )
    labels_best = clusterer.fit_predict(X_emb)

    n_clusters_best = len(set(labels_best)) - (1 if -1 in labels_best else 0)
    n_noise_best = np.sum(labels_best == -1)
    pct_noise_best = n_noise_best / len(labels_best) * 100

    display(
        df_filtered.style
        .background_gradient(cmap="RdPu", subset=["clusters"])
        .background_gradient(cmap="Greys", subset=["bruit_%"])
        .set_caption(f"HDBSCAN — Configurations dans [{target_min}, {target_max}] clusters")
    )

    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        x=X_emb[:, 0],
        y=X_emb[:, 1],
        hue=labels_best,
        palette="Spectral",
        s=20,
        alpha=0.85,
        legend=False
    )
    plt.title(
        f"HDBSCAN — {n_clusters_best} clusters ({pct_noise_best:.1f}% bruit)\n"
        f"min_cluster={best_mcs}, min_samples={best_ms}",
        color="#e75480"
    )
    plt.tight_layout()
    plt.show()

    print(f"Meilleur résultat : {n_clusters_best} clusters — {pct_noise_best:.1f}% de bruit")
    return df_filtered, best, labels_best

def summarize_clusters(df_original, labels):
    """
    Fusionne les labels de clustering avec le DataFrame d'origine
    et calcule un résumé statistique :
      - Taille de chaque cluster
      - Proportion de phishing par cluster
    """
    df_clusters = df_original.copy()
    df_clusters["cluster"] = labels

    # On ignore le bruit (-1)
    valid_clusters = [c for c in np.unique(labels) if c != -1]
    print(f"Nombre de clusters valides : {len(valid_clusters)}")

    # Synthèse par cluster
    summary = (
        df_clusters[df_clusters["cluster"] != -1]
        .groupby("cluster")
        .agg(
            Taille=("cluster", "size"),
            Part_Phishing=("Email Type", lambda x: (x == "Phishing Email").mean())
        )
        .reset_index()
        .sort_values("Taille", ascending=False)
    )

    display(
        summary.style
        .background_gradient(cmap="RdPu", subset=["Taille"])
        .background_gradient(cmap="Purples", subset=["Part_Phishing"])
        .set_caption("Répartition des clusters et proportion de phishing")
    )

    return {"df_clusters": df_clusters, "summary": summary}

def display_top_words(df_clusters, summary, top_n=10):
    """
    Affiche un tableau des mots dominants par cluster (Phishing vs Safe);
    """
    phish_clusters = summary[summary["Part_Phishing"] > 0.6]["cluster"].tolist()
    safe_clusters  = summary[summary["Part_Phishing"] < 0.4]["cluster"].tolist()

    def top_words_for_cluster(c):
        texts = df_clusters[df_clusters["cluster"] == c]["Email Text"].astype(str)
        vectorizer = CountVectorizer(stop_words="english", token_pattern=r"\b[a-zA-Z%$!?]{3,}\b")
        X_vec = vectorizer.fit_transform(texts)
        freqs = np.asarray(X_vec.sum(axis=0)).ravel()
        vocab = np.array(vectorizer.get_feature_names_out())
        top_idx = np.argsort(freqs)[::-1][:top_n]
        return ", ".join(vocab[top_idx])

    def build_table(clusters):
        data = []
        for c in clusters:
            data.append({
                "Cluster": c,
                "Top words": top_words_for_cluster(c)
            })
        return pd.DataFrame(data).sort_values("Cluster").reset_index(drop=True)

    if phish_clusters:
        df_phish = build_table(phish_clusters)
        display(
            df_phish.style
            .set_caption("Phishing Clusters — Mots les plus fréquents")
            .set_properties(subset=["Cluster"], **{"color": "#ad1457", "font-weight": "bold"})
            .set_properties(subset=["Top words"], **{"color": "#ad1457"})
            .set_table_styles([
                {"selector": "caption", "props": [("color", "#e75480"), ("font-size", "15px"), ("font-weight", "bold")]},
                {"selector": "th", "props": [("background-color", "#fce4ec"), ("color", "#ad1457"), ("font-weight", "bold")]},
                {"selector": "td", "props": [("background-color", "#fff0f5"), ("color", "#ad1457")]},
                {"selector": "table", "props": [("border", "1px solid #f8bbd0"), ("border-collapse", "collapse")]}
            ])
        )

    if safe_clusters:
        df_safe = build_table(safe_clusters)
        display(
            df_safe.style
            .set_caption("Safe Clusters — Mots les plus fréquents")
            .set_properties(subset=["Cluster"], **{"color": "#0d47a1", "font-weight": "bold"})
            .set_properties(subset=["Top words"], **{"color": "#0d47a1"})
            .set_table_styles([
                {"selector": "caption", "props": [("color", "#4a90e2"), ("font-size", "15px"), ("font-weight", "bold")]},
                {"selector": "th", "props": [("background-color", "#e3f2fd"), ("color", "#0d47a1"), ("font-weight", "bold")]},
                {"selector": "td", "props": [("background-color", "#f7fbff"), ("color", "#0d47a1")]},
                {"selector": "table", "props": [("border", "1px solid #bbdefb"), ("border-collapse", "collapse")]}
            ])
        )


def evaluate_clustering(summary_df, df_clusters):
    """
    Évalue quantitativement le clustering en comparant les étiquettes majoritaires
    de chaque cluster avec les vraies étiquettes ('Email Type').
    Calcule : précision, rappel, F1-score, accuracy globale.
    """
    # Mapping cluster -> classe majoritaire
    cluster_labels = {
        row["cluster"]: ("Phishing Email" if row["Part_Phishing"] >= 0.8 else "Safe Email")
        for _, row in summary_df.iterrows()
    }

    # Prédiction pour chaque email
    df_clusters["predicted"] = df_clusters["cluster"].map(cluster_labels)
    df_clusters = df_clusters[df_clusters["cluster"] != -1]  # ignore le bruit

    # Vraies et prédites
    y_true = df_clusters["Email Type"]
    y_pred = df_clusters["predicted"]

    # Calcul des métriques pour chaque classe
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=["Phishing Email", "Safe Email"]
    )

    acc = accuracy_score(y_true, y_pred)

    # Construction du tableau
    metrics = pd.DataFrame({
        "Classe": ["Phishing Email", "Safe Email"],
        "Précision": precision,
        "Rappel": recall,
        "F1-score": f1,
        "Support": support
    })

    metrics.loc["Global"] = ["—", acc, acc, acc, len(y_true)]

    display(metrics.style.set_caption("Évaluation du clustering HDBSCAN (corrigée)"))
    return metrics

# ================================ LLM ================================

def sample_emails_by_cluster(df_clusters: pd.DataFrame, summary: pd.DataFrame, n_per_cluster: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Tire un échantillon représentatif d'e-mails dans les clusters dominés par le phishing.
    """
    phish_clusters = summary[summary["Part_Phishing"] > 0.6]["cluster"].tolist()
    
    # On garde uniquement les e-mails des clusters phishing
    df_phish = df_clusters[df_clusters["cluster"].isin(phish_clusters)]
    
    # Échantillonnage équilibré
    sample = (
        df_phish
        .groupby("cluster", group_keys=False)
        .apply(lambda x: x.sample(n=min(n_per_cluster, len(x)), random_state=random_state))
        .reset_index(drop=True)
    )
    
    return sample

def call_ollama(prompt: str, model: str = "phi3", max_retries: int = 2, cooldown: int = 3) -> str:
    """
    Exécute un prompt localement avec Ollama (robuste et compatible Mac ARM).
    """
    for attempt in range(max_retries):
        try:
            process = subprocess.Popen(
                ["ollama", "run", model],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            output, error = process.communicate(prompt)
            
            if error and error.strip():
                print(f"Ollama stderr: {error.strip()}")
            
            if "Internal Server Error" in error or not output.strip():
                print(f"Tentative {attempt+1}/{max_retries} échouée, nouvelle tentative dans {cooldown}s...")
                time.sleep(cooldown)
                continue
            
            return output.strip()
        
        except Exception as e:
            print(f"Erreur lors de l’appel Ollama : {e}")
            time.sleep(cooldown)
    
    print("Toutes les tentatives ont échoué.")
    return ""


CATEGORIES = [
    "Banking / Credit / Loan / Insurance",
    "Investment / Crypto / Ponzi schemes",
    "Tax refund / Government payment scams",
    "Pharmaceutical / Medical / Health offers",
    "Sexual performance / Viagra / Enhancement offers",
    "Account login / Password reset / Identity theft",
    "Software download / License key / Fake update / Malware",
    "Tech support / Security alert / Fake antivirus",
    "E-commerce / Shopping scams / Payment fraud",
    "Delivery / Shipping (DHL, FedEx, UPS, La Poste, etc.)",
    "Subscription renewal / Invoicing / Billing fraud",
    "Lottery / Reward / Free gift / Contest scam",
    "Job offer / Work-from-home / Recruitment scam",
    "Romance / Dating / Adult / Explicit content",
    "Survey / Marketing / Fake promotion",
    "Commercial / Promotional offers (sales, discounts, replicas, or retail products)",
    "Generic / Social engineering / Urgent request",
    "Business transaction / Advance fee / Inheritance scams",
    "Other / Unknown"
]

def make_cluster_prompt(cluster_id: int, email_texts: list[str], max_chars_per_email: int = 150) -> str:
    """
    Crée un prompt pour classer un cluster d'e-mails en UNE catégorie dominante.
    Inclut les cas non-phishing et les domaines spécifiques observés dans le dataset.
    """
    prompt = f"""
You are a cybersecurity analyst specialized in phishing intelligence.

Task:
You are analyzing a cluster of emails (grouped by semantic similarity). 
Determine the ONE dominant category that best represents this cluster from the list below.
If none fit well, you may create a NEW category (but only if you think it's truly necessary).

Categories:
{json.dumps(CATEGORIES, indent=2)}

Output format (STRICT JSON, no extra text):
{{
  "cluster": {cluster_id},
  "category": "Chosen category from list (or new one if justified)",
  "confidence": 0.0-1.0,
  "reason": "Short summary explaining why (< 30 words)",
  "keywords": ["main indicative words (3-8)"]
}}

Guidelines:
- Choose the category that best fits the overall *intent or topic* of the emails, not just keywords.
- If none fit well, you may create a NEW category or just put Other / Unknown.
- Do NOT output explanations or reasoning outside the JSON object.

Emails (samples from this cluster):
""" + "\n\n".join([
        f"{i+1}. {txt[:max_chars_per_email].replace('\n',' ')}"
        for i, txt in enumerate(email_texts)
    ]) + "\n\nOutput:"

    return prompt.strip()



def parse_ollama_json_output(text: str):
    """Essaye de parser proprement la sortie JSON du modèle."""
    if not text or not text.strip():
        return None

    text = text.replace("```json", "").replace("```", "").strip()
    text = text.replace("True", "true").replace("False", "false").replace("None", "null")
    text = re.sub(r"(?<!\\)'", '"', text)

    if text.count("{") > text.count("}"):
        text += "}"
    if text.count("[") > text.count("]"):
        text += "]"

    try:
        obj = json.loads(text)
        if isinstance(obj, list) and obj:
            return obj[0]
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"(\{.*\})", text, re.S)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            try:
                candidate = re.sub(r"(\w+):", r'"\1":', candidate)
                return json.loads(candidate)
            except Exception:
                pass

    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def analyze_phishing_clusters_individual(df_clusters, summary,
                                         algorithm_name="hdbscan",
                                         n_per_cluster=5,
                                         model="mistral:7b-instruct",
                                         max_retries=None,
                                         cooldown=3):
    """
    Analyse chaque cluster individuellement et crée un fichier JSON par cluster.
    Les clusters déjà analysés ne sont pas relancés.
    """

    output_dir = f"data/phishing_categories/{algorithm_name}"
    os.makedirs(output_dir, exist_ok=True)

    sample = sample_emails_by_cluster(df_clusters, summary, n_per_cluster=n_per_cluster)
    clusters = sorted(sample["cluster"].unique())
    print(f"{len(clusters)} clusters à analyser ({algorithm_name.upper()})\n")

    for cluster_id in clusters:
        cluster_path = os.path.join(output_dir, f"cluster_{cluster_id}.json")

        # skip si existe
        if os.path.exists(cluster_path):
            print(f"Cluster {cluster_id} déjà traité, on passe.")
            continue

        texts = sample[sample["cluster"] == cluster_id]["Email Text"].tolist()
        if not texts:
            continue

        prompt = make_cluster_prompt(cluster_id, texts)
        print(f"Cluster {cluster_id} — {len(texts)} e-mails → {model}")

        parsed = None
        attempt = 0

        while True:
            attempt += 1
            out = call_ollama(prompt, model=model)
            parsed = parse_ollama_json_output(out)

            if parsed and isinstance(parsed, dict) and parsed.get("category"):
                print(f"Cluster {cluster_id} réussi (tentative {attempt}) : {parsed['category']}")
                break
            else:
                print(f"Parsing invalide (tentative {attempt})")

            if isinstance(max_retries, int) and attempt >= max_retries:
                print(f"Abandon après {attempt} tentatives — cluster {cluster_id}")
                parsed = {
                    "cluster": cluster_id,
                    "category": "Unknown",
                    "confidence": 0.0,
                    "reason": "Parsing failed",
                    "keywords": []
                }
                break

            time.sleep(cooldown)

        # Nettoyage numpy
        parsed = {
            k: (int(v) if isinstance(v, (np.int64, np.int32))
                else float(v) if isinstance(v, (np.float64, np.float32))
                else v)
            for k, v in parsed.items()
        }

        # Sauvegarde cluster individuel
        with open(cluster_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)

        print(f"Cluster {cluster_id} sauvegardé → {cluster_path}\n")
        time.sleep(0.8)

    print(f"\n{algorithm_name.upper()} terminé — tous les clusters sauvegardés séparément.")

def preview_cluster_emails(df_clusters: pd.DataFrame, cluster_id: int, n: int = 5, max_chars: int = 400):
    """
    Affiche un aperçu des e-mails appartenant à un cluster spécifique.
    """
    if "cluster" not in df_clusters.columns or "Email Text" not in df_clusters.columns:
        raise ValueError("Le DataFrame doit contenir les colonnes 'cluster' et 'Email Text'.")

    subset = df_clusters[df_clusters["cluster"] == cluster_id]
    if subset.empty:
        print(f"Aucun e-mail trouvé pour le cluster {cluster_id}.")
        return

    print(f"\nAperçu du cluster {cluster_id} — {len(subset)} e-mails au total\n{'='*80}")
    
    # échantillon aléatoire
    sample = subset.sample(n=min(n, len(subset)), random_state=42)

    for i, (_, row) in enumerate(sample.iterrows(), 1):
        text = str(row["Email Text"])[:max_chars].replace("\n", " ").strip()
        print(f"\nEmail {i}/{len(sample)} — Aperçu :")
        print("-"*max_chars)
        print(text if text else "(Texte vide)")
        print("-"*max_chars)

    print(f"\nFin de l’aperçu du cluster {cluster_id}.\n")

def load_llm_results(directory):
    results = []
    for f in os.listdir(directory):
        if f.endswith(".json"):
            path = os.path.join(directory, f)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    if isinstance(data, dict):
                        results.append(data)
            except Exception as e:
                print(f"Erreur lecture {f}: {e}")
    return pd.DataFrame(results)

def plot_categories_llm(X_emb, df_enriched, name):
    """
    Affiche les clusters HDBSCAN colorés par catégorie LLM.
    Le bruit (cluster = -1) est affiché en gris clair pour illustrer les points non assignés.
    """
    plt.figure(figsize=(14, 10))

    noise_mask = df_enriched["cluster"] == -1
    clean_mask = df_enriched["cluster"] != -1

    # on trace le bruit d'abord
    sns.scatterplot(
        x=X_emb[noise_mask, 0],
        y=X_emb[noise_mask, 1],
        color="lightgray",
        s=15,
        alpha=0.3,
        linewidth=0,
        label="Bruit"
    )

    sns.scatterplot(
        x=X_emb[clean_mask, 0],
        y=X_emb[clean_mask, 1],
        hue=df_enriched.loc[clean_mask, "category"].fillna("Safe Emails"),
        palette="tab20",
        s=20,
        alpha=0.85,
        linewidth=0
    )

    plt.title(
        f"Clusters {name} avec catégories LLM associées",
        fontsize=15,
        color="#e75480",
        pad=20
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles=handles,
        labels=labels,
        title="Catégories",
        bbox_to_anchor=(0.5, -0.25),
        loc="upper center",
        ncol=4, # des colonnes de 4 dans la légende
        fontsize=9,
        title_fontsize=10,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()




# ================================ Classification supervisée ================================

def run_supervised_classification(X, y, test_size=0.2, random_state=42, display_confusion=True):
    """
    Entraîne et évalue plusieurs modèles de classification supervisée (phishing vs safe).
    Compare leurs performances et affiche la matrice de confusion du meilleur modèle.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        phishing_key = "Phishing Email" if "Phishing Email" in report else "Phishing"
        results.append({
            "Modèle": name,
            "Accuracy": report["accuracy"],
            "Precision (Phishing)": report[phishing_key]["precision"],
            "Recall (Phishing)": report[phishing_key]["recall"],
            "F1-score (Phishing)": report[phishing_key]["f1-score"]
        })

    results_df = pd.DataFrame(results).sort_values("F1-score (Phishing)", ascending=False)
    print("\nRésultats comparatifs :")
    display(results_df)

    best_name = results_df.iloc[0]["Modèle"]
    best_model = models[best_name]
    y_pred_best = best_model.predict(X_test)

    if display_confusion:
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred_best, display_labels=["Safe", "Phishing"], cmap="Purples"
        )
        plt.title(f"Matrice de confusion — {best_name}")
        plt.show()

    print(f"\n→ Meilleur modèle : {best_name}")
    print(classification_report(y_test, y_pred_best))

    return results_df, best_model
