/**
 * Dashboard de Monitoring Temps Réel - EfficientNet Training
 * Gestion avancée des métriques, graphiques et alertes
 */

// Configuration du dashboard
const DASHBOARD_CONFIG = {
    updateInterval: 2000,
    maxDataPoints: 100,
    alertThresholds: {
        memoryUsage: 85,
        diskUsage: 90,
        lossStagnation: 50 // epochs without improvement
    },
    chartColors: {
        primary: '#2563eb',
        success: '#10b981',
        warning: '#f59e0b',
        danger: '#ef4444',
        info: '#06b6d4'
    }
};

// Variables globales du dashboard
let dashboardState = {
    connected: false,
    monitoring: false,
    alerts: [],
    systemMetrics: {},
    trainingMetrics: {},
    charts: {}
};

// Stockage des données historiques
let metricsHistory = {
    system: {
        cpu: [],
        memory: [],
        gpu: [],
        timestamps: []
    },
    training: {
        loss: [],
        accuracy: [],
        learningRate: [],
        epochs: []
    }
};

/**
 * Initialisation du dashboard
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Initialisation du Dashboard de Monitoring');
    
    initializeDashboard();
    setupEventListeners();
    startMonitoring();
    
    // Charger les données initiales
    loadInitialData();
});

/**
 * Initialisation des composants du dashboard
 */
function initializeDashboard() {
    initializeSystemChart();
    initializeTrainingChart();
    initializeAlertSystem();
    initializeRealTimeMetrics();
}

/**
 * Initialisation du graphique système
 */
function initializeSystemChart() {
    const ctx = document.getElementById('systemMetricsChart').getContext('2d');
    
    dashboardState.charts.system = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'CPU Usage (%)',
                    data: [],
                    borderColor: DASHBOARD_CONFIG.chartColors.info,
                    backgroundColor: 'rgba(6, 182, 212, 0.1)',
                    tension: 0.4,
                    fill: false
                },
                {
                    label: 'Memory Usage (%)',
                    data: [],
                    borderColor: DASHBOARD_CONFIG.chartColors.warning,
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    tension: 0.4,
                    fill: false
                },
                {
                    label: 'GPU Usage (%)',
                    data: [],
                    borderColor: DASHBOARD_CONFIG.chartColors.success,
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Métriques Système en Temps Réel'
                },
                legend: {
                    position: 'top'
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Temps'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Utilisation (%)'
                    },
                    min: 0,
                    max: 100
                }
            },
            animation: {
                duration: 0
            }
        }
    });
}

/**
 * Initialisation du graphique d'entraînement
 */
function initializeTrainingChart() {
    const ctx = document.getElementById('trainingMetricsChart').getContext('2d');
    
    dashboardState.charts.training = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Train Loss',
                    data: [],
                    borderColor: DASHBOARD_CONFIG.chartColors.primary,
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Val Loss',
                    data: [],
                    borderColor: DASHBOARD_CONFIG.chartColors.info,
                    backgroundColor: 'rgba(6, 182, 212, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Train Accuracy',
                    data: [],
                    borderColor: DASHBOARD_CONFIG.chartColors.success,
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                },
                {
                    label: 'Val Accuracy',
                    data: [],
                    borderColor: DASHBOARD_CONFIG.chartColors.warning,
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Métriques d\'Entraînement'
                },
                legend: {
                    position: 'top'
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Époque'
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Loss'
                    },
                    min: 0
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Accuracy (%)'
                    },
                    min: 0,
                    max: 100,
                    grid: {
                        drawOnChartArea: false
                    }
                }
            },
            animation: {
                duration: 300
            }
        }
    });
}

/**
 * Initialisation du système d'alertes
 */
function initializeAlertSystem() {
    const alertContainer = document.getElementById('alertContainer');
    if (!alertContainer) {
        console.warn('Alert container not found');
        return;
    }
    
    // Créer le conteneur d'alertes s'il n'existe pas
    alertContainer.innerHTML = `
        <div class="alert-system">
            <h6><i class="fas fa-bell me-2"></i>Alertes Système</h6>
            <div id="alertList" class="alert-list">
                <div class="text-muted small">Aucune alerte active</div>
            </div>
        </div>
    `;
}

/**
 * Initialisation des métriques temps réel
 */
function initializeRealTimeMetrics() {
    // Créer les widgets de métriques temps réel
    const metricsContainer = document.getElementById('realTimeMetrics');
    if (!metricsContainer) return;
    
    metricsContainer.innerHTML = `
        <div class="row">
            <div class="col-md-3">
                <div class="metric-widget">
                    <div class="metric-icon">
                        <i class="fas fa-microchip"></i>
                    </div>
                    <div class="metric-content">
                        <div class="metric-value" id="cpuUsage">0%</div>
                        <div class="metric-label">CPU</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-widget">
                    <div class="metric-icon">
                        <i class="fas fa-memory"></i>
                    </div>
                    <div class="metric-content">
                        <div class="metric-value" id="memoryUsage">0%</div>
                        <div class="metric-label">Memory</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-widget">
                    <div class="metric-icon">
                        <i class="fas fa-hdd"></i>
                    </div>
                    <div class="metric-content">
                        <div class="metric-value" id="diskUsage">0%</div>
                        <div class="metric-label">Disk</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-widget">
                    <div class="metric-icon">
                        <i class="fas fa-tachometer-alt"></i>
                    </div>
                    <div class="metric-content">
                        <div class="metric-value" id="trainingSpeed">0</div>
                        <div class="metric-label">Epochs/h</div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

/**
 * Configuration des écouteurs d'événements
 */
function setupEventListeners() {
    // Boutons de contrôle
    document.getElementById('pauseMonitoring')?.addEventListener('click', toggleMonitoring);
    document.getElementById('clearAlerts')?.addEventListener('click', clearAlerts);
    document.getElementById('exportMetrics')?.addEventListener('click', exportMetrics);
    
    // Gestion des graphiques
    document.getElementById('resetCharts')?.addEventListener('click', resetCharts);
    document.getElementById('toggleFullscreen')?.addEventListener('click', toggleFullscreen);
    
    // Mise à jour automatique des thresholds
    document.getElementById('alertThresholds')?.addEventListener('change', updateAlertThresholds);
}

/**
 * Démarrage du monitoring
 */
function startMonitoring() {
    dashboardState.monitoring = true;
    
    // Interval principal pour les métriques système
    setInterval(updateSystemMetrics, DASHBOARD_CONFIG.updateInterval);
    
    // Interval pour les métriques d'entraînement
    setInterval(updateTrainingMetrics, DASHBOARD_CONFIG.updateInterval);
    
    // Interval pour les alertes
    setInterval(checkAlerts, 5000);
    
    console.log('📊 Monitoring démarré');
}

/**
 * Chargement des données initiales
 */
async function loadInitialData() {
    try {
        // Charger les métriques système
        await updateSystemMetrics();
        
        // Charger l'état d'entraînement
        await updateTrainingMetrics();
        
        // Charger l'historique si disponible
        await loadTrainingHistory();
        
    } catch (error) {
        console.error('❌ Erreur lors du chargement des données initiales:', error);
    }
}

/**
 * Mise à jour des métriques système
 */
async function updateSystemMetrics() {
    try {
        const response = await fetch('/api/system/info');
        const data = await response.json();
        
        if (data.success) {
            dashboardState.systemMetrics = data.system;
            
            // Mettre à jour l'affichage
            updateSystemDisplay(data.system);
            
            // Mettre à jour les graphiques
            updateSystemChart(data.system);
            
            // Stocker dans l'historique
            storeSystemMetrics(data.system);
        }
    } catch (error) {
        console.error('❌ Erreur lors de la mise à jour des métriques système:', error);
    }
}

/**
 * Mise à jour des métriques d'entraînement
 */
async function updateTrainingMetrics() {
    try {
        const response = await fetch('/api/training/metrics');
        const data = await response.json();
        
        if (data.success) {
            dashboardState.trainingMetrics = data.metrics;
            
            // Mettre à jour l'affichage
            updateTrainingDisplay(data.metrics);
            
            // Mettre à jour les graphiques
            updateTrainingChart(data.metrics);
        }
    } catch (error) {
        console.error('❌ Erreur lors de la mise à jour des métriques d\'entraînement:', error);
    }
}

/**
 * Mise à jour de l'affichage système
 */
function updateSystemDisplay(systemInfo) {
    // CPU
    const cpuUsage = systemInfo.cpu.usage;
    document.getElementById('cpuUsage').textContent = `${cpuUsage.toFixed(1)}%`;
    
    // Memory
    const memoryUsage = systemInfo.memory.percent;
    document.getElementById('memoryUsage').textContent = `${memoryUsage.toFixed(1)}%`;
    
    // Disk
    const diskUsage = systemInfo.disk.percent;
    document.getElementById('diskUsage').textContent = `${diskUsage.toFixed(1)}%`;
    
    // GPU info (si disponible)
    if (systemInfo.gpu.available && systemInfo.gpu.devices) {
        const gpuUtil = systemInfo.gpu.devices[0].utilization || 0;
        document.getElementById('gpuUsage')?.textContent = `${gpuUtil}%`;
    }
}

/**
 * Mise à jour de l'affichage d'entraînement
 */
function updateTrainingDisplay(metrics) {
    if (!metrics.current) return;
    
    const current = metrics.current;
    
    // Métriques actuelles
    document.getElementById('currentTrainLoss')?.textContent = current.train_loss?.toFixed(4) || 'N/A';
    document.getElementById('currentValLoss')?.textContent = current.val_loss?.toFixed(4) || 'N/A';
    document.getElementById('currentTrainAcc')?.textContent = `${current.train_accuracy?.toFixed(2) || 0}%`;
    document.getElementById('currentValAcc')?.textContent = `${current.val_accuracy?.toFixed(2) || 0}%`;
    
    // Progression
    const progress = current.progress_percent || 0;
    document.getElementById('trainingProgress')?.style.width = `${progress}%`;
    document.getElementById('progressText')?.textContent = `${progress.toFixed(1)}%`;
    
    // ETA
    document.getElementById('eta')?.textContent = current.eta || 'N/A';
    
    // Vitesse d'entraînement
    if (metrics.statistics) {
        const speed = calculateTrainingSpeed(metrics.statistics);
        document.getElementById('trainingSpeed').textContent = speed.toFixed(1);
    }
}

/**
 * Mise à jour du graphique système
 */
function updateSystemChart(systemInfo) {
    const chart = dashboardState.charts.system;
    if (!chart) return;
    
    const timestamp = new Date().toLocaleTimeString();
    
    // Ajouter les nouvelles données
    chart.data.labels.push(timestamp);
    chart.data.datasets[0].data.push(systemInfo.cpu.usage);
    chart.data.datasets[1].data.push(systemInfo.memory.percent);
    chart.data.datasets[2].data.push(systemInfo.gpu.available ? 
        (systemInfo.gpu.devices?.[0]?.utilization || 0) : 0);
    
    // Limiter le nombre de points
    const maxPoints = DASHBOARD_CONFIG.maxDataPoints;
    if (chart.data.labels.length > maxPoints) {
        chart.data.labels.shift();
        chart.data.datasets.forEach(dataset => dataset.data.shift());
    }
    
    chart.update('none');
}

/**
 * Mise à jour du graphique d'entraînement
 */
function updateTrainingChart(metrics) {
    const chart = dashboardState.charts.training;
    if (!chart || !metrics.current) return;
    
    const current = metrics.current;
    const epoch = current.current_epoch;
    
    // Vérifier si c'est une nouvelle époque
    if (chart.data.labels.length === 0 || chart.data.labels[chart.data.labels.length - 1] !== epoch) {
        chart.data.labels.push(epoch);
        chart.data.datasets[0].data.push(current.train_loss);
        chart.data.datasets[1].data.push(current.val_loss);
        chart.data.datasets[2].data.push(current.train_accuracy);
        chart.data.datasets[3].data.push(current.val_accuracy);
        
        chart.update('none');
    }
}

/**
 * Stockage des métriques système
 */
function storeSystemMetrics(systemInfo) {
    const timestamp = new Date().toISOString();
    
    metricsHistory.system.timestamps.push(timestamp);
    metricsHistory.system.cpu.push(systemInfo.cpu.usage);
    metricsHistory.system.memory.push(systemInfo.memory.percent);
    metricsHistory.system.gpu.push(systemInfo.gpu.available ? 
        (systemInfo.gpu.devices?.[0]?.utilization || 0) : 0);
    
    // Limiter l'historique
    const maxHistory = 1000;
    if (metricsHistory.system.timestamps.length > maxHistory) {
        Object.keys(metricsHistory.system).forEach(key => {
            metricsHistory.system[key].shift();
        });
    }
}

/**
 * Vérification des alertes
 */
function checkAlerts() {
    const newAlerts = [];
    
    // Vérifier les seuils système
    if (dashboardState.systemMetrics.memory?.percent > DASHBOARD_CONFIG.alertThresholds.memoryUsage) {
        newAlerts.push({
            type: 'warning',
            message: `Utilisation mémoire élevée: ${dashboardState.systemMetrics.memory.percent.toFixed(1)}%`,
            timestamp: new Date()
        });
    }
    
    if (dashboardState.systemMetrics.disk?.percent > DASHBOARD_CONFIG.alertThresholds.diskUsage) {
        newAlerts.push({
            type: 'danger',
            message: `Espace disque faible: ${dashboardState.systemMetrics.disk.percent.toFixed(1)}%`,
            timestamp: new Date()
        });
    }
    
    // Vérifier la stagnation de l'entraînement
    if (dashboardState.trainingMetrics.statistics?.epochs_without_improvement > DASHBOARD_CONFIG.alertThresholds.lossStagnation) {
        newAlerts.push({
            type: 'warning',
            message: 'Stagnation détectée dans l\'entraînement',
            timestamp: new Date()
        });
    }
    
    // Ajouter les nouvelles alertes
    newAlerts.forEach(alert => addAlert(alert));
}

/**
 * Ajout d'une alerte
 */
function addAlert(alert) {
    dashboardState.alerts.push(alert);
    
    // Limiter le nombre d'alertes
    if (dashboardState.alerts.length > 20) {
        dashboardState.alerts.shift();
    }
    
    updateAlertDisplay();
    
    // Notification du navigateur
    if ('Notification' in window && Notification.permission === 'granted') {
        new Notification('NightScan Training Alert', {
            body: alert.message,
            icon: '/static/icon.png'
        });
    }
}

/**
 * Mise à jour de l'affichage des alertes
 */
function updateAlertDisplay() {
    const alertList = document.getElementById('alertList');
    if (!alertList) return;
    
    if (dashboardState.alerts.length === 0) {
        alertList.innerHTML = '<div class="text-muted small">Aucune alerte active</div>';
        return;
    }
    
    const html = dashboardState.alerts.map(alert => `
        <div class="alert alert-${alert.type} alert-sm">
            <i class="fas fa-exclamation-triangle me-2"></i>
            ${alert.message}
            <small class="text-muted ms-2">${alert.timestamp.toLocaleTimeString()}</small>
        </div>
    `).join('');
    
    alertList.innerHTML = html;
}

/**
 * Calcul de la vitesse d'entraînement
 */
function calculateTrainingSpeed(statistics) {
    if (!statistics.avg_epoch_time) return 0;
    return 3600 / statistics.avg_epoch_time; // epochs par heure
}

/**
 * Chargement de l'historique d'entraînement
 */
async function loadTrainingHistory() {
    try {
        const response = await fetch('/api/training/history?limit=100');
        const data = await response.json();
        
        if (data.success && data.history.length > 0) {
            // Remplir le graphique avec l'historique
            const chart = dashboardState.charts.training;
            
            data.history.forEach(entry => {
                chart.data.labels.push(entry.current_epoch);
                chart.data.datasets[0].data.push(entry.train_loss);
                chart.data.datasets[1].data.push(entry.val_loss);
                chart.data.datasets[2].data.push(entry.train_accuracy);
                chart.data.datasets[3].data.push(entry.val_accuracy);
            });
            
            chart.update();
        }
    } catch (error) {
        console.error('❌ Erreur lors du chargement de l\'historique:', error);
    }
}

/**
 * Fonctions utilitaires
 */

function toggleMonitoring() {
    dashboardState.monitoring = !dashboardState.monitoring;
    const btn = document.getElementById('pauseMonitoring');
    btn.textContent = dashboardState.monitoring ? 'Pause' : 'Resume';
    btn.className = dashboardState.monitoring ? 'btn btn-warning' : 'btn btn-success';
}

function clearAlerts() {
    dashboardState.alerts = [];
    updateAlertDisplay();
}

function resetCharts() {
    Object.values(dashboardState.charts).forEach(chart => {
        chart.data.labels = [];
        chart.data.datasets.forEach(dataset => {
            dataset.data = [];
        });
        chart.update();
    });
}

function exportMetrics() {
    const data = {
        system: metricsHistory.system,
        training: dashboardState.trainingMetrics,
        alerts: dashboardState.alerts,
        exported_at: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `nightscan_metrics_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    
    URL.revokeObjectURL(url);
}

function toggleFullscreen() {
    const dashboardContainer = document.getElementById('dashboardContainer');
    if (dashboardContainer.requestFullscreen) {
        dashboardContainer.requestFullscreen();
    }
}

function updateAlertThresholds() {
    const form = document.getElementById('alertThresholds');
    if (form) {
        const formData = new FormData(form);
        DASHBOARD_CONFIG.alertThresholds.memoryUsage = parseInt(formData.get('memoryThreshold'));
        DASHBOARD_CONFIG.alertThresholds.diskUsage = parseInt(formData.get('diskThreshold'));
    }
}

// Demander la permission pour les notifications
if ('Notification' in window && Notification.permission === 'default') {
    Notification.requestPermission();
}

// Export des fonctions pour utilisation externe
window.dashboardAPI = {
    addAlert,
    updateSystemMetrics,
    updateTrainingMetrics,
    exportMetrics,
    toggleMonitoring
};

console.log('📊 Dashboard monitoring system loaded');