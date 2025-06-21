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

The Flask application stores results in the `prediction` table of its MySQL database. Each entry has a `result` field in JSON format. Its `predictions` array contains the best species in the first element. To populate WordPress, simply copy these records into `ns_predictions`. A small Python script can act as a bridge:

```python
import MySQLdb  # or pymysql

flask_db = MySQLdb.connect(host='localhost', user='appuser', passwd='secret', db='nightscan')
wordpress = MySQLdb.connect(host='localhost', user='wpuser', passwd='secret', db='wordpress')

with flask_db.cursor() as cur_src, wordpress.cursor() as cur_dest:
    cur_src.execute(
        "SELECT user_id, JSON_EXTRACT(result, '$.predictions[0].label') AS species FROM prediction"
    )
    for row in cur_src.fetchall():
        cur_dest.execute(
            "INSERT INTO wp_ns_predictions (user_id, species, predicted_at) VALUES (%s, %s, NOW())",
            (row[0], row[1])
        )
    wordpress.commit()
```

Adjust the connection parameters and fields to match your exact schema. You can run this via `cron` for regular synchronization.

## Plugin installation

1. Copy the `prediction-charts` folder into `wp-content/plugins/`.
2. Activate **Prediction Charts** from the WordPress admin.
3. Insert the `[nightscan_chart]` shortcode into a page or post.

Once activated, each logged‑in user will see a chart showing the number of predictions per hour and species for their own data.

## Audio upload plugin

The repository also ships with **NightScan Audio Upload**. Copy the
`audio-upload` folder to `wp-content/plugins/` and activate it. Place the
`[nightscan_uploader]` shortcode in a page to show the upload form. The
plugin reads the API URL from the `ns_api_endpoint` option, which you can
set for example with:

```bash
wp option update ns_api_endpoint https://your-vps.example/api/predict
```

The API may reside on a different server than WordPress. If so, ensure
that CORS is allowed and use HTTPS when required so the browser is able to
post the files successfully.

Uploads are subject to the same limits as the Flask form: each WAV file
may be up to 100\u00a0MB and a given user cannot store more than
10\u00a0GB in total.

WordPress itself can block large uploads if PHP is configured with small
limits. Check the `upload_max_filesize` and `post_max_size` values in your
`php.ini`. Both must be set to at least `100M` so the plugin can accept
files up to 100 MB.

Uploaded files must be WAV audio. The plugin verifies the MIME type using
`finfo_file` or `wp_check_filetype` and rejects any other format before it
sends the data to the API endpoint.
