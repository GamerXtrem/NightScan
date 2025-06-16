<?php
/*
Plugin Name: Prediction Charts
Description: Display prediction counts per hour and species using Chart.js.
Version: 1.0
*/

if (!defined('ABSPATH')) exit;

function pc_enqueue_scripts() {
    // Load Chart.js with a scheme-relative URL so the plugin works on HTTP and HTTPS
    wp_enqueue_script('chartjs', '//cdn.jsdelivr.net/npm/chart.js', array(), null, true);
}
add_action('wp_enqueue_scripts', 'pc_enqueue_scripts');

function pc_get_data() {
    global $wpdb, $current_user;
    wp_get_current_user();
    $table = $wpdb->prefix . 'ns_predictions';
    $rows = $wpdb->get_results($wpdb->prepare(
        "SELECT species, DATE_FORMAT(predicted_at, '%Y-%m-%d %H:00:00') as hour, COUNT(*) as cnt FROM $table WHERE user_id = %d GROUP BY species,hour ORDER BY hour", $current_user->ID), ARRAY_A);

    $data = array();
    foreach ($rows as $r) {
        $hour = $r['hour'];
        if (!isset($data[$hour])) $data[$hour] = array();
        $data[$hour][$r['species']] = intval($r['cnt']);
    }
    return $data;
}

function pc_render_chart() {
    if (!is_user_logged_in()) return '<p>You must be logged in to view this chart.</p>';
    $data = pc_get_data();
    $hours = array_keys($data);
    sort($hours);
    $species = array();
    foreach ($data as $h => $vals) {
        foreach ($vals as $sp => $count) {
            if (!in_array($sp, $species)) $species[] = $sp;
        }
    }
    sort($species);
    $dataset_js = '';
    foreach ($species as $sp) {
        $counts = array();
        foreach ($hours as $h) {
            $counts[] = isset($data[$h][$sp]) ? $data[$h][$sp] : 0;
        }
        $color = sprintf('#%06X', mt_rand(0, 0xFFFFFF));
        $dataset_js .= json_encode(array('label'=>$sp,'data'=>$counts,'backgroundColor'=>$color)) . ',';
    }
    $dataset_js = rtrim($dataset_js, ',');
    ob_start();
    ?>
    <canvas id="pc-chart" width="400" height="200"></canvas>
    <script>
    const ctx = document.getElementById('pc-chart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: <?php echo json_encode($hours); ?>,
            datasets: [<?php echo $dataset_js; ?>]
        },
        options: { scales: { x: { stacked: true }, y: { stacked: true } } }
    });
    </script>
    <?php
    return ob_get_clean();
}
add_shortcode('nightscan_chart', 'pc_render_chart');
