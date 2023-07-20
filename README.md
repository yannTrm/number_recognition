# number_recognition

Le projet vise à mettre en œuvre un système de reconnaissance de chiffres écrits à la main en utilisant le modèle de réseau de neurones convolutionnels LeNet-4. Le modèle pré-entraîné est chargé à partir d'un fichier, et les utilisateurs peuvent fournir une image contenant un ou plusieurs chiffres écrits à la main. Le système segmente automatiquement les chiffres présents dans l'image, les traite et fait des prédictions sur chaque chiffre individuellement. Les prédictions sont renvoyées sous forme de résultats numériques. Le projet utilise la bibliothèque OpenCV pour le traitement d'image et TensorFlow pour le modèle de réseau de neurones. Il offre une solution simple et efficace pour la reconnaissance des chiffres manuscrits et peut être utilisé dans diverses applications telles que la reconnaissance de codes postaux sur des enveloppes, l'automatisation de la lecture de factures, etc.

## Table des matières

- [Aperçu](#aperçu)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure des Fichiers](#structure-des-fichiers)
- [Crédits](#crédits)

## Aperçu

Le projet vise à mettre en œuvre un système de reconnaissance de chiffres écrits à la main en utilisant le modèle de réseau de neurones convolutionnels LeNet-4. Le modèle pré-entraîné est chargé à partir d'un fichier, et les utilisateurs peuvent fournir une image contenant un ou plusieurs chiffres écrits à la main. Le système segmente automatiquement les chiffres présents dans l'image, les traite et fait des prédictions sur chaque chiffre individuellement. Les prédictions sont renvoyées sous forme de résultats numériques. Le projet utilise la bibliothèque OpenCV pour le traitement d'image et TensorFlow pour le modèle de réseau de neurones. Il offre une solution simple et efficace pour la reconnaissance des chiffres manuscrits et peut être utilisé dans diverses applications telles que la reconnaissance de codes postaux sur des enveloppes, l'automatisation de la lecture de factures, etc.

## Installation

1. Cloner le dépôt :

```bash
git clone https://github.com/yannTrm/number_recognition
cd votre_projet
```

Installer les dépendances requises :

```bash
pip install -r requirements.txt
```

Télécharger le modèle pré-entraîné model.h5 et placez-le dans le répertoire model/.

Utilisation


`main.py`

Ce script est le point d'entrée du projet et sert d'exemple pour utiliser les fonctions fournies par les autres modules.

```python
# Importer les modules nécessaires
import process_image as seg
import leNet4 as LeNet4

# Constantes
PATH_MODEL = "./model/model.h5"
FILE_DATA = "chemin_vers_vos_données"

# Charger le modèle pré-entraîné
model = LeNet4.load_single_model(PATH_MODEL)

# Traiter l'image et obtenir les prédictions
predictions = seg.process_image(model, FILE_DATA)

# Afficher les prédictions
print(predictions)
```

`process_image.py`
Ce module contient des fonctions pour traiter les images et faire des prédictions en utilisant le modèle pré-entraîné.

load_image(chemin_image)
Cette fonction charge une image depuis le chemin de fichier chemin_image en utilisant OpenCV. Elle renvoie l'image chargée sous forme d'un tableau NumPy en niveaux de gris.

grey_to_binary(image, threshold=127)
Cette fonction applique un seuillage pour convertir une image en niveaux de gris en une image binaire. Les pixels avec des valeurs supérieures au seuil seront définis à 255 (blanc), et les pixels avec des valeurs inférieures ou égales au seuil seront définis à 0 (noir).

get_roi(data, location, roi="rois")
Cette fonction extrait une région d'intérêt (ROI) à partir du dictionnaire data en fonction des paramètres location et roi. Elle renvoie les coordonnées (x, y), la largeur et la hauteur de la ROI.

create_roi(image, x, y, width, height)
Cette fonction crée une région d'intérêt (ROI) à partir de l'image donnée en utilisant les coordonnées (x, y), la largeur et la hauteur spécifiées. Elle renvoie la ROI sous forme d'une sous-image de l'image d'origine.

segment_image(image)
Cette fonction segmente une image en objets individuels et renvoie une liste des objets segmentés. Elle utilise la détection des contours et filtre les objets qui ne répondent pas aux critères définis dans la fonction is_digit_segment.

is_digit_segment(segment)
Cette fonction vérifie si un segment donné cont



