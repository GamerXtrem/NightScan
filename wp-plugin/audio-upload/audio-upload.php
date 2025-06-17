<?php
/*
Plugin Name: NightScan Audio Upload
Description: Provides a shortcode allowing users to upload WAV files that are forwarded to a configurable API endpoint.
Version: 1.0
*/

if (!defined('ABSPATH')) exit;

function nsau_render_form() {
    $output = '';
    if ($_SERVER['REQUEST_METHOD'] === 'POST' && !empty($_FILES['ns_audio_file'])) {
        if (!isset($_POST['nsau_nonce']) || !wp_verify_nonce($_POST['nsau_nonce'], 'nsau_upload')) {
            $output .= '<p>Nonce verification failed.</p>';
        } else {
            $file = $_FILES['ns_audio_file'];
            if ($file['error'] !== UPLOAD_ERR_OK) {
                $output .= '<p>File upload error.</p>';
            } else {
                $endpoint = get_option('ns_api_endpoint');
                if (!$endpoint) {
                    $output .= '<p>API endpoint not configured.</p>';
                } else {
                    $body = file_get_contents($file['tmp_name']);
                    $response = wp_remote_post($endpoint, array(
                        'headers' => array('Content-Type' => 'audio/wav'),
                        'body'    => $body,
                    ));
                    if (is_wp_error($response)) {
                        $output .= '<p>Request error: '.esc_html($response->get_error_message()).'</p>';
                    } else {
                        $json = wp_remote_retrieve_body($response);
                        $output .= '<pre>'.esc_html($json).'</pre>';
                    }
                }
            }
        }
    }
    $output .= '<form method="post" enctype="multipart/form-data">';
    $output .= '<input type="file" name="ns_audio_file" accept=".wav" required>';
    $output .= wp_nonce_field('nsau_upload', 'nsau_nonce', true, false);
    $output .= '<input type="submit" value="Upload">';
    $output .= '</form>';
    return $output;
}
add_shortcode('nightscan_uploader', 'nsau_render_form');
