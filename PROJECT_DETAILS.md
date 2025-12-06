# Description du Projet : Vision Thermique AI

##  Objectif
Ce projet vise à démocratiser l'accès à la thermographie intelligente en utilisant la Computer Vision et l'Intelligence Artificielle. Il permet d'estimer la température d'objets en temps réel à partir d'un flux vidéo standard ou d'une caméra thermique, en couplant la détection d'objets (YOLOv8) avec des modèles d'estimation thermique.

## Architecture Technique

Le système repose sur trois piliers principaux :

1.  **Détection d'Objets (YOLOv8)**
    *   Utilisation du modèle YOLOv8 Nano pour une performance optimale en temps réel.
    *   Identification automatique des personnes, objets (tasses, bouteilles, etc.) dans le champ de vision.
    *   Capacité de filtrer les classes d'objets pertinentes pour l'analyse thermique.

2.  **Estimation de Température (Hybride)**
    *   **Mode Simulation (Heuristique)** : Pour les démonstrations et le développement sans matériel coûteux. Utilise des connaissances a priori (ex: température corporelle humaine ~37°C) avec modélisation de bruit réaliste.
    *   **Mode Régression (IA)** : Architecture prête à accueillir un modèle de Deep Learning entraîné pour prédire la température réelle à partir des caractéristiques visuelles (nécessite un dataset RGB-Thermique).

3.  **Interface & Calibration**
    *   Affichage en surimpression (HUD) des données critiques.
    *   Outils de calibration en direct (Offset, Émissivité) pour ajuster les mesures aux conditions environnementales.

##  Cas d'Usage Potentiels

*   **Santé & Sécurité** : Détection rapide de fièvre dans les foules (avec caméra thermique réelle).
*   **Industrie** : Surveillance de surchauffe sur des machines ou équipements électriques.
*   **Domotique** : Détection de présence et confort thermique.
*   **Éducation** : Apprentissage de la vision par ordinateur et de la fusion de capteurs.

##  Évolutions Futures

*   Intégration native des SDK de caméras thermiques (FLIR, Seek Thermal).
*   Entraînement d'un modèle de régression robuste sur un dataset propriétaire.
*   Alertes automatiques (SMS/Email) en cas de dépassement de seuil de température.
