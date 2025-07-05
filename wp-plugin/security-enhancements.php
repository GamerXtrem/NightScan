<?php
/*
Plugin Name: NightScan Security Enhancements
Description: Additional security measures for NightScan WordPress plugins
Version: 1.0
*/

if (!defined('ABSPATH')) exit;

/**
 * Security Enhancement Class for NightScan WordPress Plugins
 */
class NightScan_Security {
    
    private static $instance = null;
    private $failed_attempts = array();
    private $max_attempts = 5;
    private $lockout_duration = 900; // 15 minutes
    
    public static function getInstance() {
        if (self::$instance === null) {
            self::$instance = new self();
        }
        return self::$instance;
    }
    
    private function __construct() {
        add_action('init', array($this, 'init_security'));
        add_action('wp_login_failed', array($this, 'record_failed_login'));
        add_filter('authenticate', array($this, 'check_login_attempts'), 30, 3);
        add_action('wp_ajax_nightscan_upload', array($this, 'secure_ajax_upload'));
        add_action('wp_ajax_nopriv_nightscan_upload', array($this, 'deny_anonymous_upload'));
    }
    
    /**
     * Initialize security measures
     */
    public function init_security() {
        // Add security headers
        $this->add_security_headers();
        
        // Sanitize all input data
        $this->sanitize_global_input();
        
        // Block suspicious requests
        $this->block_suspicious_requests();
        
        // Clean up old failed attempts
        $this->cleanup_old_attempts();
    }
    
    /**
     * Add security headers
     */
    private function add_security_headers() {
        if (!headers_sent()) {
            header('X-Content-Type-Options: nosniff');
            header('X-Frame-Options: SAMEORIGIN');
            header('X-XSS-Protection: 1; mode=block');
            header('Referrer-Policy: strict-origin-when-cross-origin');
            
            // Only add HSTS if we're on HTTPS
            if (isset($_SERVER['HTTPS']) && $_SERVER['HTTPS'] === 'on') {
                header('Strict-Transport-Security: max-age=31536000; includeSubDomains');
            }
        }
    }
    
    /**
     * Sanitize global input data
     */
    private function sanitize_global_input() {
        // Sanitize $_GET
        if (!empty($_GET)) {
            foreach ($_GET as $key => $value) {
                $_GET[$key] = $this->deep_sanitize($value);
            }
        }
        
        // Sanitize $_POST (except for file uploads)
        if (!empty($_POST)) {
            foreach ($_POST as $key => $value) {
                if ($key !== 'ns_audio_file') { // Don't sanitize file upload data
                    $_POST[$key] = $this->deep_sanitize($value);
                }
            }
        }
        
        // Sanitize $_COOKIE
        if (!empty($_COOKIE)) {
            foreach ($_COOKIE as $key => $value) {
                $_COOKIE[$key] = $this->deep_sanitize($value);
            }
        }
    }
    
    /**
     * Deep sanitization of data
     */
    private function deep_sanitize($data) {
        if (is_array($data)) {
            return array_map(array($this, 'deep_sanitize'), $data);
        } else {
            // Remove null bytes
            $data = str_replace(chr(0), '', $data);
            
            // Remove control characters except tab, newline, and carriage return
            $data = preg_replace('/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/', '', $data);
            
            // Basic sanitization
            return sanitize_text_field($data);
        }
    }
    
    /**
     * Block suspicious requests
     */
    private function block_suspicious_requests() {
        $user_agent = isset($_SERVER['HTTP_USER_AGENT']) ? $_SERVER['HTTP_USER_AGENT'] : '';
        $request_uri = isset($_SERVER['REQUEST_URI']) ? $_SERVER['REQUEST_URI'] : '';
        
        // Block common attack patterns
        $suspicious_patterns = array(
            // SQL injection attempts
            'union.*select',
            'concat.*\(',
            'information_schema',
            
            // XSS attempts
            '<script',
            'javascript:',
            'onload=',
            'onerror=',
            
            // File inclusion attempts
            '\.\./\.\.',
            'etc/passwd',
            'proc/self',
            
            // Command injection
            'system\(',
            'exec\(',
            'shell_exec\(',
            
            // Common malicious user agents
            'sqlmap',
            'nikto',
            'nmap',
            'masscan',
        );
        
        foreach ($suspicious_patterns as $pattern) {
            if (preg_match('/' . $pattern . '/i', $user_agent . $request_uri)) {
                $this->log_security_event('Suspicious request blocked', array(
                    'pattern' => $pattern,
                    'user_agent' => $user_agent,
                    'request_uri' => $request_uri,
                    'ip' => $this->get_client_ip()
                ));
                
                wp_die('Access denied.', 'Security Error', array('response' => 403));
            }
        }
    }
    
    /**
     * Record failed login attempts
     */
    public function record_failed_login($username) {
        $ip = $this->get_client_ip();
        $current_time = time();
        
        if (!isset($this->failed_attempts[$ip])) {
            $this->failed_attempts[$ip] = array();
        }
        
        $this->failed_attempts[$ip][] = $current_time;
        
        // Keep only recent attempts
        $this->failed_attempts[$ip] = array_filter(
            $this->failed_attempts[$ip],
            function($timestamp) use ($current_time) {
                return ($current_time - $timestamp) < $this->lockout_duration;
            }
        );
        
        // Store in database for persistence
        update_option('nightscan_failed_attempts', $this->failed_attempts);
        
        $this->log_security_event('Failed login attempt', array(
            'username' => sanitize_user($username),
            'ip' => $ip,
            'attempt_count' => count($this->failed_attempts[$ip])
        ));
    }
    
    /**
     * Check login attempts before authentication
     */
    public function check_login_attempts($user, $username, $password) {
        $ip = $this->get_client_ip();
        
        // Load stored attempts
        $stored_attempts = get_option('nightscan_failed_attempts', array());
        if (is_array($stored_attempts)) {
            $this->failed_attempts = $stored_attempts;
        }
        
        if (isset($this->failed_attempts[$ip])) {
            $recent_attempts = count($this->failed_attempts[$ip]);
            
            if ($recent_attempts >= $this->max_attempts) {
                $this->log_security_event('Login blocked due to too many attempts', array(
                    'ip' => $ip,
                    'attempt_count' => $recent_attempts
                ));
                
                return new WP_Error(
                    'too_many_attempts',
                    sprintf(
                        'Too many failed login attempts. Please try again in %d minutes.',
                        round($this->lockout_duration / 60)
                    )
                );
            }
        }
        
        return $user;
    }
    
    /**
     * Secure AJAX upload handler
     */
    public function secure_ajax_upload() {
        // Verify nonce
        if (!wp_verify_nonce($_POST['nonce'], 'nightscan_ajax_upload')) {
            wp_die('Security check failed.', 'Security Error', array('response' => 403));
        }
        
        // Verify user capabilities
        if (!current_user_can('upload_files')) {
            wp_die('Insufficient permissions.', 'Security Error', array('response' => 403));
        }
        
        // Rate limiting
        $user_id = get_current_user_id();
        $last_upload = get_user_meta($user_id, 'last_upload_time', true);
        $current_time = time();
        
        if ($last_upload && ($current_time - $last_upload) < 60) { // 1 minute rate limit
            wp_send_json_error('Rate limit exceeded. Please wait before uploading again.');
            return;
        }
        
        update_user_meta($user_id, 'last_upload_time', $current_time);
        
        // Additional file validation would go here
        // For now, just log the secure upload attempt
        $this->log_security_event('Secure AJAX upload attempt', array(
            'user_id' => $user_id,
            'ip' => $this->get_client_ip()
        ));
        
        wp_send_json_success('Upload security check passed.');
    }
    
    /**
     * Deny anonymous upload attempts
     */
    public function deny_anonymous_upload() {
        $this->log_security_event('Anonymous upload attempt blocked', array(
            'ip' => $this->get_client_ip()
        ));
        
        wp_die('Authentication required.', 'Security Error', array('response' => 401));
    }
    
    /**
     * Get client IP address
     */
    private function get_client_ip() {
        $ip_headers = array(
            'HTTP_CF_CONNECTING_IP',     // Cloudflare
            'HTTP_CLIENT_IP',            // Proxy
            'HTTP_X_FORWARDED_FOR',      // Load balancer/proxy
            'HTTP_X_FORWARDED',          // Proxy
            'HTTP_X_CLUSTER_CLIENT_IP',  // Cluster
            'HTTP_FORWARDED_FOR',        // Proxy
            'HTTP_FORWARDED',            // Proxy
            'REMOTE_ADDR'                // Standard
        );
        
        foreach ($ip_headers as $header) {
            if (!empty($_SERVER[$header])) {
                $ip = $_SERVER[$header];
                
                // Handle comma-separated IPs (X-Forwarded-For)
                if (strpos($ip, ',') !== false) {
                    $ip = trim(explode(',', $ip)[0]);
                }
                
                // Validate IP address
                if (filter_var($ip, FILTER_VALIDATE_IP, FILTER_FLAG_NO_PRIV_RANGE | FILTER_FLAG_NO_RES_RANGE)) {
                    return $ip;
                }
            }
        }
        
        return isset($_SERVER['REMOTE_ADDR']) ? $_SERVER['REMOTE_ADDR'] : 'unknown';
    }
    
    /**
     * Clean up old failed attempts
     */
    private function cleanup_old_attempts() {
        $current_time = time();
        $cleaned = false;
        
        foreach ($this->failed_attempts as $ip => $attempts) {
            $this->failed_attempts[$ip] = array_filter(
                $attempts,
                function($timestamp) use ($current_time) {
                    return ($current_time - $timestamp) < $this->lockout_duration;
                }
            );
            
            if (empty($this->failed_attempts[$ip])) {
                unset($this->failed_attempts[$ip]);
                $cleaned = true;
            }
        }
        
        if ($cleaned) {
            update_option('nightscan_failed_attempts', $this->failed_attempts);
        }
    }
    
    /**
     * Log security events
     */
    private function log_security_event($event, $data = array()) {
        $log_entry = array(
            'timestamp' => current_time('mysql'),
            'event' => $event,
            'data' => $data,
            'user_id' => get_current_user_id(),
            'session_id' => session_id()
        );
        
        // Store in WordPress options (for small scale) or custom table (for larger scale)
        $security_log = get_option('nightscan_security_log', array());
        
        // Keep only last 1000 entries to prevent database bloat
        if (count($security_log) >= 1000) {
            $security_log = array_slice($security_log, -999);
        }
        
        $security_log[] = $log_entry;
        update_option('nightscan_security_log', $security_log);
        
        // Also log to WordPress error log if enabled
        if (defined('WP_DEBUG_LOG') && WP_DEBUG_LOG) {
            error_log('NightScan Security: ' . $event . ' - ' . json_encode($data));
        }
    }
    
    /**
     * Get security log (for admin dashboard)
     */
    public function get_security_log($limit = 100) {
        $security_log = get_option('nightscan_security_log', array());
        return array_slice($security_log, -$limit);
    }
    
    /**
     * Validate file upload security
     */
    public static function validate_upload_security($file) {
        // File size check
        if ($file['size'] > 100 * 1024 * 1024) { // 100MB
            return new WP_Error('file_too_large', 'File exceeds maximum size limit.');
        }
        
        // MIME type check
        $allowed_types = array('audio/wav', 'audio/x-wav');
        if (!in_array($file['type'], $allowed_types)) {
            return new WP_Error('invalid_file_type', 'Only WAV files are allowed.');
        }
        
        // File extension check
        $file_info = pathinfo($file['name']);
        if (!isset($file_info['extension']) || strtolower($file_info['extension']) !== 'wav') {
            return new WP_Error('invalid_extension', 'File must have .wav extension.');
        }
        
        // Filename security check
        if (preg_match('/[^a-zA-Z0-9._-]/', $file['name'])) {
            return new WP_Error('invalid_filename', 'Filename contains invalid characters.');
        }
        
        // File content validation (basic header check)
        if (!self::validate_wav_file_content($file['tmp_name'])) {
            return new WP_Error('invalid_content', 'File content is not a valid WAV file.');
        }
        
        return true;
    }
    
    /**
     * Validate WAV file content
     */
    private static function validate_wav_file_content($file_path) {
        $handle = fopen($file_path, 'rb');
        if (!$handle) {
            return false;
        }
        
        // Read first 44 bytes (standard WAV header)
        $header = fread($handle, 44);
        fclose($handle);
        
        if (strlen($header) < 44) {
            return false;
        }
        
        // Check RIFF signature
        if (substr($header, 0, 4) !== 'RIFF') {
            return false;
        }
        
        // Check WAVE format
        if (substr($header, 8, 4) !== 'WAVE') {
            return false;
        }
        
        // Check fmt chunk
        if (substr($header, 12, 4) !== 'fmt ') {
            return false;
        }
        
        return true;
    }
}

// Initialize security enhancements
NightScan_Security::getInstance();

/**
 * Security dashboard for administrators
 */
if (is_admin()) {
    add_action('admin_menu', 'nightscan_security_admin_menu');
    
    function nightscan_security_admin_menu() {
        add_options_page(
            'NightScan Security',
            'NightScan Security',
            'manage_options',
            'nightscan-security',
            'nightscan_security_admin_page'
        );
    }
    
    function nightscan_security_admin_page() {
        if (!current_user_can('manage_options')) {
            wp_die('Unauthorized access.');
        }
        
        $security = NightScan_Security::getInstance();
        $log_entries = $security->get_security_log(50);
        
        echo '<div class="wrap">';
        echo '<h1>NightScan Security Dashboard</h1>';
        
        echo '<h2>Recent Security Events</h2>';
        echo '<table class="wp-list-table widefat fixed striped">';
        echo '<thead><tr><th>Timestamp</th><th>Event</th><th>Details</th><th>User ID</th></tr></thead>';
        echo '<tbody>';
        
        foreach (array_reverse($log_entries) as $entry) {
            echo '<tr>';
            echo '<td>' . esc_html($entry['timestamp']) . '</td>';
            echo '<td>' . esc_html($entry['event']) . '</td>';
            echo '<td>' . esc_html(json_encode($entry['data'])) . '</td>';
            echo '<td>' . esc_html($entry['user_id']) . '</td>';
            echo '</tr>';
        }
        
        echo '</tbody>';
        echo '</table>';
        echo '</div>';
    }
}

// Hook into existing upload functions to add security
add_filter('wp_handle_upload_prefilter', function($file) {
    $validation = NightScan_Security::validate_upload_security($file);
    if (is_wp_error($validation)) {
        $file['error'] = $validation->get_error_message();
    }
    return $file;
});
?>