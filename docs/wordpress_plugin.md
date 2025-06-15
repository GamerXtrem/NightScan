# Plugin WordPress "Prediction Charts"

Ce document décrit la structure de la table `ns_predictions` utilisée par le plugin
`prediction-charts` ainsi que la synchronisation possible des données
entre la base Flask et la base WordPress.

## Structure de la table `ns_predictions`

Le plugin s'attend à trouver une table dans la base de données WordPress
(portant le préfixe habituel, par exemple `wp_ns_predictions`) contenant au
minimum les colonnes suivantes :

| Colonne      | Type        | Description                                    |
| ------------ | ----------- | ---------------------------------------------- |
| `id`         | INT AUTO_INCREMENT | Clef primaire. |
| `user_id`    | INT NOT NULL | Identifiant de l'utilisateur WordPress. |
| `species`    | VARCHAR(100) NOT NULL | Nom de l'espèce prédite. |
| `predicted_at` | DATETIME NOT NULL | Date et heure de la prédiction. |

D'autres champs (score de confiance, nom du fichier, etc.) peuvent être
ajoutés selon vos besoins mais ne sont pas utilisés par le shortcode.

## Export des prédictions depuis Flask

L'application Flask stocke les résultats dans la table `prediction` de la
base `site.db` (SQLite par défaut). Chaque enregistrement possède un champ
`result` au format JSON contenant un tableau `predictions` dont le premier
élément représente la meilleure espèce trouvée. Pour alimenter WordPress il
suffit de copier ces enregistrements dans `ns_predictions`. Un petit script
Python peut faire le relais :

```python
import sqlite3
import MySQLdb  # ou pymysql

sqlite_conn = sqlite3.connect('web/site.db')
mysql = MySQLdb.connect(host='localhost', user='wpuser', passwd='secret', db='wordpress')

for row in sqlite_conn.execute(
    "SELECT user_id, json_extract(result, '$.predictions[0].label') AS species, id FROM prediction"
):
    with mysql.cursor() as cur:
        cur.execute(
            "INSERT INTO wp_ns_predictions (user_id, species, predicted_at) VALUES (%s, %s, NOW())",
            (row[0], row[1])
        )
    mysql.commit()
```

Adaptez les paramètres de connexion et les champs selon votre schéma exact.
L'opération peut être programmée via `cron` pour une synchronisation régulière.

## Installation du plugin

1. Copiez le dossier `prediction-charts` dans `wp-content/plugins/`.
2. Activez **Prediction Charts** depuis l'administration WordPress.
3. Insérez le shortcode `[nightscan_chart]` dans une page ou un article.

Une fois activé, chaque utilisateur connecté verra un graphique représentant
le nombre de prédictions par heure et par espèce sur les données qui lui
sont associées.
