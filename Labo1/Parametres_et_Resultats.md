# Configuration de l'Expérience et Résultats

Ce fichier récapitule les paramètres utilisés dans le code pour votre rapport.

## 1. Paramètres du GAN

| Paramètre | Valeur Utilisée | Pourquoi ? |
| :--- | :--- | :--- |
| **Dataset** | Fashion-MNIST | Plus complexe que MNIST (chiffres), permet de mieux voir les différences. |
| **Batch Size** | 128 | Compromis standard pour la stabilité du gradient. |
| **Z_DIM** (Dimension Latente) | 64 | Suffisant pour encoder la variété des vêtements sans être trop complexe. |
| **Learning Rate** | 0.0002 (2e-4) | Valeur classique pour Adam avec les GANs (évite l'instabilité). |
| **Loss Function** | Hinge Loss | Plus stable que la BCE (Binary Cross Entropy) pour éviter le "Mode Collapse". |
| **Époques** | 5 | Suffisant pour voir des résultats sur GPU (augmenter si nécessaire). |

### Résultats GAN (Données Expérimentales)

| Configuration | FID Score (Plus bas = Mieux) | Observation |
| :--- | :--- | :--- |
| **Z=32, LR=2e-4** | 0.1478 | Score moyen, probablement manque de capacité. |
| **Z=64, LR=2e-4** | **0.0657** | **Meilleur résultat !** Bon équilibre. |
| **Z=128, LR=2e-4** | 0.0799 | Légèrement moins bon, peut-être trop complexe pour 5 époques. |
| **Z=64, LR=1e-4** | 0.1151 | Apprentissage trop lent avec ce LR réduit. |
| **Batch=64** | 0.0716 | Bon résultat, mais le Batch 128 reste supérieur (0.0657). |

*Conclusion* : La configuration optimale semble être **Z_DIM=64** avec un **Learning Rate de 0.0002** et un **Batch Size de 128**.

---

## 2. Paramètres du VAE

| Paramètre | Valeur Utilisée | Pourquoi ? |
| :--- | :--- | :--- |
| **Latent Dim** | 16 | Assez petit pour forcer la compression, assez grand pour garder les détails. |
| **Learning Rate** | 0.002 (2e-3) | Plus élevé que le GAN car le VAE est plus stable. |
| **Loss Function** | L1 + KL | L1 donne des images moins floues que MSE (Mean Squared Error). |
| **Époques** | 5 | Rapide à entraîner. |

### Résultats VAE (Données Expérimentales)
*   **Observations** : Le VAE ne calcule pas de FID par défaut dans ce script (N/A), mais on peut observer la qualité des reconstructions.
*   **Impact de Latent Dim** :
    *   **L=8** : Probablement flou (compression forte).
    *   **L=16 / 32** : Meilleure définition attendue.


---

## 3. Comparaison Finale

| Modèle | Avantages | Inconvénients | Score FID |
| :--- | :--- | :--- | :--- |
| **GAN** | Images plus nettes | Entraînement instable | ... |
| **VAE** | Entraînement stable, bonne diversité | Images floues | ... |
