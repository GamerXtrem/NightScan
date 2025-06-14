# "Prediction Charts" WordPress Plugin

This document describes the `ns_predictions` table used by the `prediction-charts` plugin and how to sync data between the Flask database and WordPress.

## Structure of the `ns_predictions` table

The plugin expects a table in the WordPress database (with the usual prefix, e.g. `wp_ns_predictions`) containing at least the following columns:

| Column      | Type        | Description                         |
| ----------- | ----------- | ----------------------------------- |
| `id`        | INT AUTO_INCREMENT | Primary key. |
| `user_id`   | INT NOT NULL | WordPress user ID. |
| `species`   | VARCHAR(100) NOT NULL | Name of the predicted species. |
| `predicted_at` | DATETIME NOT NULL | Date and time of the prediction. |

You may add other fields (confidence score, file name, etc.) as needed, but they are not used by the shortcode.

## Exporting predictions from Flask

The Flask application stores results in the `prediction` table of the `site.db` (SQLite by default). Each entry has a `result` field in JSON format. Its `predictions` array contains the best species in the first element. To populate WordPress, simply copy these records into `ns_predictions`. A small Python script can act as a bridge:

```python
import sqlite3
import MySQLdb  # or pymysql

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

Adjust the connection parameters and fields to match your exact schema. You can run this via `cron` for regular synchronization.

## Plugin installation

1. Copy the `prediction-charts` folder into `wp-content/plugins/`.
2. Activate **Prediction Charts** from the WordPress admin.
3. Insert the `[nightscan_chart]` shortcode into a page or post.

Once activated, each logged‑in user will see a chart showing the number of predictions per hour and species for their own data.
