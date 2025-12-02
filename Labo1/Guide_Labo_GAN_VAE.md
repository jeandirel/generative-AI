# Guide du Labo : Comprendre, Ajuster et Réussir (GAN vs VAE)

Ce document a pour but de vous expliquer clairement les objectifs du TP, les concepts clés, et surtout **comment gérer les paramètres** comme demandé par votre professeur.

## 1. Comprendre les Concepts (Sans Jargon)

### Le GAN (Generative Adversarial Network)
Imaginez un jeu du chat et de la souris :
*   **Le Générateur (Le Faussaire)** : Essaie de créer de faux billets (images) pour tromper la police. Il ne voit jamais les vrais billets, il apprend uniquement grâce aux retours du policier.
*   **Le Discriminateur (Le Policier)** : Essaie de distinguer les vrais billets des faux.
*   **L'objectif** : À la fin, le faussaire est devenu si bon que le policier ne peut plus faire la différence (50% de chance de se tromper).

### Le VAE (Variational Autoencoder)
Imaginez un système de compression ultra-intelligent :
*   **L'Encodeur** : Prend une image et la résume en quelques nombres (le "code" ou vecteur latent).
*   **Le Décodeur** : Prend ces quelques nombres et essaie de reconstruire l'image originale.
*   **La subtilité** : On force ces "nombres" à suivre une distribution normale (une courbe en cloche). Cela permet de générer de nouvelles images en prenant des nombres au hasard dans cette distribution.

---

## 2. Les Paramètres à Ajuster (Ce que le prof attend)

Le professeur vous demande d'ajuster des paramètres pour observer leur impact. Voici les plus critiques et pourquoi :

### A. La Dimension Latente (`Z_DIM` pour GAN, `LATENT` pour VAE)
C'est la taille du vecteur de bruit en entrée (ex: 16, 32, 64, 128).
*   **Le Problème** :
    *   **Trop petit (ex: 2 ou 8)** : Le modèle n'a pas assez de "place" pour stocker toutes les variations des chiffres (forme, inclinaison, épaisseur). Les images seront pauvres ou se ressembleront toutes.
    *   **Trop grand (ex: 512)** : Le modèle a trop de liberté, l'apprentissage peut être plus lent ou difficile à structurer.
*   **Ce qu'il faut faire** : Le notebook suggère de tester **32** et **128**. Comparez la qualité des images.

### B. Le Learning Rate (`LR`)
La vitesse à laquelle le modèle apprend.
*   **Le Problème (Surtout pour le GAN)** :
    *   **Trop haut** : Le GAN est **instable**. Le discriminateur apprend trop vite et "écrase" le générateur, ou le générateur oscille et ne produit que du bruit.
    *   **Trop bas** : L'apprentissage est interminable.
*   **Ce qu'il faut faire** : Si votre GAN ne génère rien de bon (images grises ou bruit statique), essayez de **diviser le LR par 2 ou 10** (ex: passer de `2e-4` à `1e-4`).

### C. Le Batch Size (`BATCH`)
Le nombre d'images vues avant de mettre à jour le modèle.
*   **Impact** : Un batch plus grand (128, 256) donne une estimation plus stable de l'erreur, ce qui aide le GAN. Un batch trop petit rend l'entraînement chaotique.

---

## 3. Problèmes Récurrents et Solutions

Voici les problèmes classiques que vous allez rencontrer et comment les expliquer/contourner :

### Problème 1 : "Mode Collapse" (GAN)
*   **Symptôme** : Le générateur produit **toujours la même image** (ex: toujours le chiffre "1"), peu importe le bruit en entrée.
*   **Pourquoi ?** : Le générateur a trouvé une image qui trompe bien le discriminateur et se contente de la répéter à l'infini (solution de facilité).
*   **Solution** :
    *   Utiliser un **Learning Rate plus petit**.
    *   Vérifier que le discriminateur n'est pas trop puissant par rapport au générateur.
    *   La "Hinge Loss" (utilisée dans ce TP) aide à combattre ce problème par rapport à la perte classique (BCE).

### Problème 2 : Instabilité / Non-convergence (GAN)
*   **Symptôme** : Les courbes de perte (loss) font des montagnes russes et ne descendent jamais vraiment. Les images générées changent radicalement d'une époque à l'autre sans s'améliorer.
*   **Pourquoi ?** : C'est la nature du GAN (équilibre instable).
*   **Solution** :
    *   C'est souvent normal. Regardez les images plutôt que les courbes.
    *   Si les images restent du bruit : **Réduisez le Learning Rate**.
    *   Assurez-vous que les données sont bien normalisées entre **[-1, 1]** (c'est fait dans le code fourni).

### Problème 3 : Images Floues (VAE)
*   **Symptôme** : Les images générées par le VAE sont un peu floues, moins nettes que celles du GAN.
*   **Pourquoi ?** : C'est intrinsèque au VAE. Il utilise une perte de reconstruction (L1 ou MSE) qui fait une "moyenne", ce qui lisse les détails.
*   **Solution** :
    *   Utiliser une perte **L1** au lieu de MSE (déjà fait dans le code).
    *   Accepter que le VAE est meilleur pour capturer la structure globale (diversité) que les détails fins (netteté), contrairement au GAN.

## Résumé pour votre Rapport
Pour impressionner le prof, montrez que vous avez compris ces compromis :
1.  **GAN** : Images plus nettes, mais difficile à entraîner (instable, mode collapse).
2.  **VAE** : Entraînement stable, bonne diversité, mais images un peu floues.
3.  **Paramètres** : Montrez des comparaisons (ex: "Avec Z_DIM=8, les chiffres sont tous pareils, avec Z_DIM=64, ils sont variés").
