#!/usr/bin/env python3
"""Copy predictions from the Flask database to WordPress."""

import argparse
import os
from urllib.parse import urlparse

import pymysql


def parse_mysql_dsn(dsn: str) -> dict:
    """Parse ``mysql://user:pass@host/db`` and return connection kwargs."""
    r = urlparse(dsn)
    if r.scheme != "mysql":
        raise ValueError("Unsupported DSN scheme: %s" % r.scheme)
    return {
        "host": r.hostname or "localhost",
        "user": r.username or "",
        "password": r.password or "",
        "database": r.path.lstrip("/"),
        "port": r.port or 3306,
        "charset": "utf8mb4",
    }


def export_predictions(flask_dsn: str, wp_dsn: str, table: str) -> None:
    """Copy prediction rows from the Flask DB to the WordPress table."""
    src = pymysql.connect(**parse_mysql_dsn(flask_dsn))
    dest = pymysql.connect(**parse_mysql_dsn(wp_dsn))
    with src.cursor() as cur_src, dest.cursor() as cur_dest:
        cur_src.execute(
            "SELECT user_id, JSON_EXTRACT(result, '$.predictions[0].label') "
            "AS species FROM prediction"
        )
        for user_id, species in cur_src.fetchall():
            cur_dest.execute(
                f"INSERT INTO {table} (user_id, species, predicted_at) "
                "VALUES (%s, %s, NOW())",
                (user_id, species),
            )
    dest.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export predictions to WordPress")
    parser.add_argument(
        "--flask-dsn",
        default=os.getenv("FLASK_DB_URI"),
        help="mysql://user:pass@host/db for the Flask app",
    )
    parser.add_argument(
        "--wp-dsn",
        default=os.getenv("WORDPRESS_DB_URI"),
        help="mysql://user:pass@host/db for WordPress",
    )
    parser.add_argument(
        "--table",
        default=os.getenv("WP_TABLE", "wp_ns_predictions"),
        help="WordPress table name",
    )
    args = parser.parse_args()
    if not args.flask_dsn or not args.wp_dsn:
        parser.error("Both --flask-dsn and --wp-dsn must be provided")
    export_predictions(args.flask_dsn, args.wp_dsn, args.table)


if __name__ == "__main__":
    main()
