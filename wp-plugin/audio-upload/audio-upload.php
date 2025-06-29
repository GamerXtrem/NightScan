<?php
/*
Plugin Name: NightScan Audio Upload
Description: Provides a shortcode allowing users to upload WAV files that are forwarded to a configurable API endpoint.
Version: 1.0
*/

if (!defined('ABSPATH')) exit;

// Maximum storage per user (10 GB)
if (!defined('NSAU_MAX_TOTAL_BYTES')) {
    define('NSAU_MAX_TOTAL_BYTES', 10 * 1024 * 1024 * 1024);
}

function nsau_render_form() {
    $output = '';

    if (!is_user_logged_in()) {
        return '<p>You must be logged in to upload files.</p>';
    }

    $user_id = get_current_user_id();
    $total = intval(get_user_meta($user_id, 'nsau_total_bytes', true));

    if ($_SERVER['REQUEST_METHOD'] === 'POST' && !empty($_FILES['ns_audio_file'])) {
        if (!isset($_POST['nsau_nonce']) || !wp_verify_nonce($_POST['nsau_nonce'], 'nsau_upload')) {
            $output .= '<p>Nonce verification failed.</p>';
        } else {
            $file = $_FILES['ns_audio_file'];
            if ($file['error'] !== UPLOAD_ERR_OK) {
                $output .= '<p>File upload error.</p>';
            } elseif ($file['size'] > 100 * 1024 * 1024) {
                $output .= '<p>File exceeds 100 MB limit.</p>';
            } elseif ($total + $file['size'] > NSAU_MAX_TOTAL_BYTES) {
                $output .= '<p>Upload quota exceeded (10 GB total).</p>';
            } else {
                $mime = '';
                if (function_exists('finfo_open')) {
                    $finfo = finfo_open(FILEINFO_MIME_TYPE);
                    if ($finfo) {
                        $mime = finfo_file($finfo, $file['tmp_name']);
                        finfo_close($finfo);
                    }
                }
                if (!$mime && function_exists('wp_check_filetype')) {
                    $ft = wp_check_filetype($file['name']);
                    if (!empty($ft['type'])) {
                        $mime = $ft['type'];
                    }
                }
                if ($mime !== 'audio/wav') {
                    $output .= '<p>Invalid file type. WAV required.</p>';
                } else {
                    $endpoint = get_option('ns_api_endpoint');
                    if (!$endpoint) {
                        $output .= '<p>API endpoint not configured.</p>';
                    } else {
                        $endpoint = trim($endpoint);
                        $validated = function_exists('wp_http_validate_url') ? wp_http_validate_url($endpoint) : filter_var($endpoint, FILTER_VALIDATE_URL);
                        $scheme = parse_url($endpoint, PHP_URL_SCHEME);
                        if (!$validated || strtolower($scheme) !== 'https') {
                            $output .= '<p>Invalid or non-HTTPS API endpoint.</p>';
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
                                update_user_meta($user_id, 'nsau_total_bytes', $total + $file['size']);
                            }
                        }
                    }
                }
            }
        }
    }
    $remaining_gb = max(0, (NSAU_MAX_TOTAL_BYTES - $total) / (1024 * 1024 * 1024));
    $output .= '<form method="post" enctype="multipart/form-data">';
    $output .= '<input type="file" name="ns_audio_file" accept=".wav" required>';
    $output .= '<p>Maximum 100 MB per file, 10 GB total for your account. WAV files only.</p>';
    $output .= '<p>Remaining quota: '.number_format($remaining_gb, 2).' GB</p>';
    $output .= wp_nonce_field('nsau_upload', 'nsau_nonce', true, false);
    $output .= '<input type="submit" value="Upload">';
    $output .= '</form>';
    $output .= '<script>document.addEventListener("DOMContentLoaded",function(){var i=document.querySelector("input[name=ns_audio_file]");if(i){i.addEventListener("change",function(){if(this.files.length&&this.files[0].size>104857600){alert("File exceeds 100 MB limit.");this.value="";}});}});</script>';
    return $output;
}
add_shortcode('nightscan_uploader', 'nsau_render_form');
