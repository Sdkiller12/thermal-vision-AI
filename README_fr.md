# Thermal Vision AI (Vision Thermique IA)

Une application de vision par ordinateur basée sur Python qui estime la température des objets détectés en utilisant YOLOv8 et des modèles Heuristiques/Régression.

## Fonctionnalités
- **Détection d'Objets** : Utilise YOLOv8 (Nano) pour une détection temps réel.
- **Estimation de Température** :
  - **Mode Simulation** : Estime la température basée sur la classe de l'objet (ex: Personne ~37°C) avec un bruit réaliste.
  - **Mode Régression** : Emplacement pour un modèle de régression RGB-vers-Thermique entraîné.
- **Interface Temps Réel** : Affiche les boîtes englobantes, les étiquettes de température et un HUD (Affichage Tête Haute).
- **Calibration** : Curseurs pour ajuster l'offset de température et l'émissivité en temps réel.

## Installation

1. **Prérequis** : Python 3.8+
2. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```
   *Note : Cela installera `opencv-python`, `ultralytics`, `torch`, etc.*

## Utilisation

### Lancer avec la Webcam (Mode Simulation)
```bash
python main.py
```

### Lancer avec un Fichier Vidéo
```bash
python main.py --source chemin/vers/video.mp4
```

### Lancer en Mode Régression
```bash
python main.py --mode regression
```

### Contrôles
- **Curseurs (Trackbars)** :
  - `Calibration Offset` : Ajuste l'offset de température de base (+/- 50°C).
  - `Emissivity` : Ajuste le facteur d'émissivité (0.0 - 1.0).
- **Clavier** :
  - Appuyez sur `q` pour quitter.

## Structure du Projet
- `src/` : Code source.
  - `detector.py` : Wrapper YOLOv8.
  - `temperature.py` : Logique d'estimation de la température.
  - `video_stream.py` : Gestionnaire de capture vidéo.
  - `config.py` : Paramètres de configuration.
- `models/` : Répertoire pour les poids des modèles.
- `data/` : Répertoire pour les datasets/vidéos.
- `train_temperature_model.py` : Script squelette pour l'entraînement d'un modèle de régression personnalisé.

## Entraîner un Modèle Personnalisé (FLIR ADAS)
Le projet inclut désormais le support pour l'entraînement d'un modèle de régression de température utilisant le **Dataset FLIR ADAS**.

### 1. Configuration du Dataset
Le script d'entraînement attend la structure du dataset FLIR ADAS dans le répertoire `data/` :
- `data/images_thermal_train/analyticsData` : Contient les images thermiques TIFF 16-bit.
- `data/images_thermal_train/coco.json` : Contient les annotations COCO pour les objets.

**Note** : Les datasets RGB et Thermiques fournis se sont avérés non appariés (IDs vidéo disjoints), donc le pipeline est actuellement configuré pour un **Entraînement Thermique Uniquement** (prédire la température max à partir de l'image thermique elle-même).

### 2. Lancer l'Entraînement
Pour entraîner le modèle :
```bash
python train_temperature_model.py
```
Cela va :
- Analyser les annotations COCO pour extraire les découpes d'objets.
- Normaliser les données thermiques 16-bit.
- Entraîner un régresseur CNN pendant 5 époques.
- Sauvegarder le modèle entraîné dans `models/thermal_regressor.pth`.

lien pour telecharger les data : https://adas-dataset-v2.flirconservator.com/#downloadguide
