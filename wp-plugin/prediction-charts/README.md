# Prediction Charts WordPress Plugin

This plugin provides a shortcode `[nightscan_chart]` that displays a stacked bar chart of predictions per hour and species. Data is read from the `ns_predictions` table and filtered so that logged‑in users see only their own results.

Usage:
1. Upload the `prediction-charts` folder to your `wp-content/plugins/` directory.
2. Activate the plugin in WordPress.
3. Insert the shortcode on any page: `[nightscan_chart]`.

The chart is built with Chart.js loaded from a CDN.
