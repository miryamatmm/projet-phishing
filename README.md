# Clustering d’e-mails de phishing

## Description
Ce projet a pour objectif d’analyser un jeu de données d’e-mails réels (Safe / Phishing) à l’aide de **méthodes non supervisées** de *clustering sémantique*.  
L’étude repose sur des **embeddings textuels (all-MiniLM-L6-v2)** et sur des techniques de **réduction de dimension** (*t-SNE*, *UMAP*), avant l’application de plusieurs algorithmes de regroupement (**KMeans**, **DBSCAN**, **HDBSCAN**).  
Une étape complémentaire combine **HDBSCAN** avec un **modèle de langage (LLM Mistral-7B)** pour catégoriser automatiquement les types de phishing détectés.

## Ressources
- **Dataset Kaggle** : [Phishing Emails Dataset](https://www.kaggle.com/datasets/subhajournal/phishingemails)  
- **Rapport complet (PDF)** : [Télécharger ici](https://drive.google.com/file/d/1RIeg_QJmuq0cYCihDlWwCO0feRhHIpk4/view?usp=sharing)

## Contributions

Le projet a été réalisé en collaboration par quatre étudiantes du Master Informatique à l’Université Claude Bernard Lyon 1.


| **Tâches principales** | **Miryam Atamna** | **Olivia Chen** | **Imane Gara** | **Niama Chibani** |
|--------------------------|:----------------:|:----------------:|:----------------:|:----------------:|
| Recherche des données | X | XXX | X |  |
| Nettoyage et préparation du jeu de données | XX | X | XXX |  |
| Vectorisation sémantique (embeddings MiniLM) | X | XXX | X | X |
| Réduction de dimension (UMAP / t-SNE) | XX | XX | XX | X |
| Clustering (KMeans, DBSCAN, HDBSCAN) | XXX | X | XX | X |
| Interprétation sémantique (LLM Mistral-7B) | XXX | XX | XX | X |
| Visualisation et analyse des résultats |X | X | X | X |
| Classification supervisée et comparaison | X |  |  | XXX |
| Rédaction du rapport et synthèse finale | X | X | X | X |



## 🎀 Auteurs
Projet réalisé dans le cadre du Master 2 Data Science — Université Claude Bernard Lyon 1  
**Miryam Atamna**, **Imane Gara**, **Olivia Chen**, **Niama Chibani**  
*Encadrant : Rémy Cazabet — Octobre 2025*
