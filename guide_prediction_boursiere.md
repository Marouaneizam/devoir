# GRAND GUIDE : ANATOMIE D'UN PROJET DE PRÉDICTION BOURSIÈRE

Ce document décortique chaque étape du cycle de vie d'un projet de
Machine Learning appliqué à la finance. Il est conçu pour passer du
niveau *débutant qui copie du code* au niveau *ingénieur qui comprend
les mécanismes internes*.

------------------------------------------------------------------------

## 1. Contexte Métier et Mission

### 🔍 Le Problème (Business Case)

Sur les marchés financiers, la volatilité et le volume d'informations
rendent la prise de décision humaine difficile.

**Objectif :** Créer un *Assistant IA* pour détecter la tendance future
du marché (Hausse ou Baisse).

### ⚠️ L'Enjeu Critique : Matrice des Gains et Pertes

-   **Faux Positif (Achat à tort)** : IA → Hausse, Réel → Baisse →
    *Perte d'argent*\
-   **Faux Négatif (Occasion manquée)** : IA → Baisse, Réel → Hausse →
    *Manque à gagner*

👉 L'IA doit privilégier **la Précision** pour protéger le capital.

### 📊 Les Données

Dataset : `Market_Trend_External.csv`

-   **X (Features)** : indicateurs (prix, volume, volatilité, sentiment,
    VIX...)
-   **y (Target)** :
    -   `1` = Hausse\
    -   `0` = Baisse/Neutre

------------------------------------------------------------------------

## 2. Code Python (Laboratoire)

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Market_Trend_External.csv')
df['target'] = (df['Daily_Return_Pct'] > 0).astype(int)

np.random.seed(42)
df_dirty = df.copy()
cols_to_corrupt = [c for c in df.columns if c not in ['Date', 'target', 'Daily_Return_Pct']]
for col in cols_to_corrupt:
    df_dirty.loc[df_dirty.sample(frac=0.05).index, col] = np.nan

X = df_dirty[cols_to_corrupt]
y = df_dirty['target']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_clean = pd.DataFrame(X_imputed, columns=X.columns)

print(X_clean[['Close_Price', 'Volume', 'Sentiment_Score']].describe())

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Baisse','Hausse']))

plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Baisse','Hausse'], yticklabels=['Baisse','Hausse'])
plt.title('Matrice de Confusion')
plt.show()
```

------------------------------------------------------------------------

## 3. Analyse Approfondie : Nettoyage (Data Wrangling)

### ❗ Le Problème Mathématique du NaN

Les algorithmes ne peuvent pas gérer les valeurs manquantes.

### 🧠 Mécanique de l'Imputation

`SimpleImputer(strategy='mean')` : - **fit** : calcule la moyenne -
**transform** : remplace les trous par cette moyenne

### ⚠️ Le Coin de l'Expert : Look-Ahead Bias

Ne jamais utiliser des statistiques calculées avec des données futures !

------------------------------------------------------------------------

## 4. Analyse Exploratoire (EDA)

### 📌 Interpréter `.describe()`

-   **Mean vs Médiane** : Volume souvent asymétrique (jours extrêmes)
-   **Std** : volatilité du marché\
-   **Multicollinéarité** : prix (High/Low/Close) ≈ 99% corrélés

------------------------------------------------------------------------

## 5. Méthodologie (Split)

### 🎯 Objectif : Généralisation

Ne pas mémoriser le passé, mais apprendre les mécanismes.

### Paramètres

-   `test_size=0.2` → ratio 80/20
-   `random_state=42` → reproductibilité

### ⚠️ Séries temporelles :

En finance, **on ne mélange jamais les jours** (pas de shuffle).

------------------------------------------------------------------------

## 6. Focus Théorique : Random Forest 🌲

### 🌳 A. Faiblesse de l'Arbre unique

Trop sensible au bruit du marché.

### 🌲 B. Force du Groupe (Bagging)

-   Diversité des données
-   Diversité des features

### 🗳️ C. Consensus

Majorité des votes = décision finale

------------------------------------------------------------------------

## 7. Évaluation (L'Heure de Vérité)

### 🔢 Matrice de Confusion

-   **TP** : Hausse → Hausse (gain)
-   **TN** : Baisse → Baisse (protection)
-   **FP** : Hausse → Baisse (perte réelle)
-   **FN** : Baisse → Hausse (opportunité manquée)

### 🎯 Métriques Clés

-   **Accuracy** : souvent trompeuse
-   **Precision** : qualité du signal (prioritaire en trading)
-   **Recall** : capacité à capturer les hausses

------------------------------------------------------------------------

## 🏁 Conclusion du Projet

Ce projet démontre que prédire la bourse n'est pas qu'une question de
code.\
Il faut : - un nettoyage rigoureux, - un modèle robuste (Random
Forest), - une évaluation orientée gestion du risque.

👉 **La Précision prime sur l'Accuracy en trading.**
