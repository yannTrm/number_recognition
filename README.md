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

Cette fonction vérifie si un segment donné contient un chiffre. Elle calcule le rapport de pixels non nuls dans le segment et applique un seuil pour déterminer s'il représente un chiffre. Elle renvoie True s'il est probablement un chiffre et False sinon.

process_image(model, path_data=FILE_DATA, result_file=None, boost=False)

Cette fonction traite une image en la chargeant, en créant une région d'intérêt, en segmentant l'image et en faisant des prédictions en utilisant le modèle fourni. Si boost est True, elle utilise la fonction predict_boosted de leNet4.py, sinon elle utilise la fonction predict. La fonction renvoie les prédictions.

handle_value_error(e)

Cette fonction gère l'exception ValueError levée pendant le traitement de l'image et renvoie -1.

leNet4.py

Ce module contient des fonctions liées au modèle LeNet-4, y compris le chargement du modèle pré-entraîné et les prédictions.

load_single_model(model_path)

Cette fonction charge le modèle LeNet-4 pré-entraîné à partir du chemin de fichier model_path. Elle renvoie le modèle chargé.

predict(model, data)

Cette fonction prend le modèle chargé model et les données d'entrée data pour faire des prédictions. Elle renvoie les prédictions pour les données fournies.

predict_boosted(model, data)

Cette fonction est une alternative à predict, spécialement conçue pour les modèles boostés (si applicable). Elle prend le modèle chargé model et les données d'entrée data pour faire des prédictions. Elle renvoie les prédictions pour les données fournies.
