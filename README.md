# Projet IA 1.03

Ceci est notre projet 1A Pôle IA à CentraleSupélec : Etude de l'activité physique d'utilisateurs de smartphones

## Mise en place de l'environnement et librairies utilisées

Les librairies Python utilisées dans ce projet sont précisées dans le fichier "requirements.txt". 
Pour initialiser l'environnement avec les bonnes librairies : "pip install -r requirments.txt"

## Structure du dossier GitHub

### Les données utilisées

Le jeu de données utilisé est disponible au lien suivant :
[Lien du jeu de données](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

Les données sont dans le dossier **original_data**

### Le dossier utils

Les fonctions de score sont dans **utils.scoring**

Les fichiers **io.py**, **merging.py** et **tools.py** contiennent des fonctions appelées fréquemment dans tous les autres codes.

### Les prétraitements

Les prétraitements sont accessibles dans les jupyter notebooks :
* dtw.ipynb
* fft.ipynb
* cnn.ipynb

Les sorties éventuelles de ces prétraitements se trouvent dans le fichier **outputs**. Le fichier **saved_models** contient des sauvegardes des paramètres du CNN.
Chaque prétraitement est également sous forme d'un fichier python. **fft.ipynb** utilise des fonctions de **fourier_analysis.py**.

Par Adrien Berger, Raffael Bolla Di Lorenzo, Fabien Charlier, Ian-Evan Michel, Aymeric Palaric