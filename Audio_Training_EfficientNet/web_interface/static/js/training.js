/**
 * JavaScript pour l'Interface d'Entraînement EfficientNet
 * Gère les WebSockets, graphiques temps réel, et interactions utilisateur
 */

// Configuration globale
const CONFIG = {
    socketURL: window.location.origin,
    updateInterval: 1000,
    maxLogEntries: 100,
    chartUpdateInterval: 500
};

// Variables globales
let socket = null;
let metricsChart = null;
let trainingHistory = {
    epochs: [],
    trainLoss: [],
    valLoss: [],
    trainAccuracy: [],
    valAccuracy: []
};

// État de l'application
let appState = {
    connected: false,
    training: false,
    sessionId: null,
    startTime: null
};

/**
 * Initialisation de l'application
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Initialisation de l\'interface d\'entraînement EfficientNet');
    
    initializeSocket();
    initializeChart();
    loadSystemInfo();
    setupEventListeners();
    
    // Charger le statut initial
    loadTrainingStatus();
});

/**
 * Configuration des WebSockets
 */
function initializeSocket() {
    console.log('🔌 Connexion WebSocket...');
    
    socket = io(CONFIG.socketURL);
    
    socket.on('connect', function() {
        console.log('✅ WebSocket connecté');
        updateConnectionStatus(true);
        socket.emit('join_training');
    });
    
    socket.on('disconnect', function() {
        console.log('❌ WebSocket déconnecté');
        updateConnectionStatus(false);
    });
    
    socket.on('training_progress', function(data) {
        console.log('📊 Mise à jour progression:', data);
        updateTrainingProgress(data);
    });
    
    socket.on('training_log', function(logEntry) {
        console.log('📝 Nouveau log:', logEntry);
        addLogEntry(logEntry);
    });
    
    socket.on('training_complete', function(data) {
        console.log('🎉 Entraînement terminé:', data);
        onTrainingComplete(data);
    });
    
    socket.on('training_status', function(data) {
        console.log('📋 Statut d\'entraînement:', data);
        updateUIState(data);
    });
}

/**
 * Mise à jour du statut de connexion
 */
function updateConnectionStatus(connected) {
    appState.connected = connected;
    const statusEl = document.getElementById('connectionStatus');
    const textEl = document.getElementById('connectionText');
    
    if (connected) {
        statusEl.className = 'status-indicator status-active';
        textEl.textContent = 'Connecté';
    } else {
        statusEl.className = 'status-indicator status-inactive';
        textEl.textContent = 'Déconnecté';
    }
}

/**
 * Initialisation du graphique des métriques
 */
function initializeChart() {
    const ctx = document.getElementById('metricsChart').getContext('2d');
    
    metricsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Train Loss',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Val Loss',
                    data: [],
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Train Accuracy',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                },
                {
                    label: 'Val Accuracy',
                    data: [],
                    borderColor: '#f59e0b',
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
                    text: 'Évolution des Métriques d\'Entraînement'
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
                        drawOnChartArea: false,
                    },
                }
            },
            animation: {
                duration: 750,
                easing: 'easeInOutQuart'
            }
        }
    });
}

/**
 * Mise à jour du graphique avec nouvelles données
 */
function updateChart(progress) {
    if (!metricsChart || !progress) return;
    
    const epoch = progress.current_epoch;
    
    // Ajouter les nouvelles données
    trainingHistory.epochs.push(epoch);
    trainingHistory.trainLoss.push(progress.train_loss);
    trainingHistory.valLoss.push(progress.val_loss);
    trainingHistory.trainAccuracy.push(progress.train_accuracy);
    trainingHistory.valAccuracy.push(progress.val_accuracy);
    
    // Mettre à jour le graphique
    metricsChart.data.labels = trainingHistory.epochs;
    metricsChart.data.datasets[0].data = trainingHistory.trainLoss;
    metricsChart.data.datasets[1].data = trainingHistory.valLoss;
    metricsChart.data.datasets[2].data = trainingHistory.trainAccuracy;
    metricsChart.data.datasets[3].data = trainingHistory.valAccuracy;
    
    metricsChart.update('none'); // Animation rapide
}

/**
 * Chargement des informations système
 */
async function loadSystemInfo() {
    try {
        const response = await fetch('/api/system_info');
        const data = await response.json();
        
        updateSystemInfoDisplay(data);
    } catch (error) {
        console.error('❌ Erreur lors du chargement des infos système:', error);
    }
}

/**
 * Mise à jour de l'affichage des informations système
 */
function updateSystemInfoDisplay(data) {
    const container = document.getElementById('systemInfo');
    
    let html = '';
    
    // GPU Info
    if (data.gpu && data.gpu.available) {
        html += `
            <div class="system-metric">
                <span><i class="fas fa-microchip text-primary"></i> GPU</span>
                <span class="badge bg-success">Disponible</span>
            </div>
        `;
        
        if (data.gpu.device_name) {
            html += `
                <div class="system-metric">
                    <span>Modèle GPU</span>
                    <span>${data.gpu.device_name}</span>
                </div>
            `;
        }
    } else {
        html += `
            <div class="system-metric">
                <span><i class="fas fa-microchip text-warning"></i> GPU</span>
                <span class="badge bg-warning">Non disponible</span>
            </div>
        `;
    }
    
    // CPU Info
    html += `
        <div class="system-metric">
            <span><i class="fas fa-processor text-info"></i> CPU</span>
            <span>${data.cpu.count} cœurs (${data.cpu.usage}%)</span>
        </div>
    `;
    
    // Memory Info
    const memoryPercent = data.memory.percent;
    const memoryColor = memoryPercent > 80 ? 'danger' : memoryPercent > 60 ? 'warning' : 'success';
    html += `
        <div class="system-metric">
            <span><i class="fas fa-memory text-${memoryColor}"></i> Mémoire</span>
            <span>${memoryPercent.toFixed(1)}% utilisée</span>
        </div>
    `;
    
    // Disk Info
    const diskPercent = data.disk.percent;
    const diskColor = diskPercent > 90 ? 'danger' : diskPercent > 70 ? 'warning' : 'success';
    html += `
        <div class="system-metric">
            <span><i class="fas fa-hdd text-${diskColor}"></i> Disque</span>
            <span>${diskPercent.toFixed(1)}% utilisé</span>
        </div>
    `;
    
    container.innerHTML = html;
}

/**
 * Configuration des écouteurs d'événements
 */
function setupEventListeners() {
    // Actualisation automatique des infos système
    setInterval(loadSystemInfo, 30000); // Toutes les 30 secondes
}

/**
 * Chargement du statut d'entraînement actuel
 */
async function loadTrainingStatus() {
    try {
        const response = await fetch('/api/training_status');
        const data = await response.json();
        
        updateUIState(data);
    } catch (error) {
        console.error('❌ Erreur lors du chargement du statut:', error);
    }
}

/**
 * Mise à jour de l'état de l'interface utilisateur
 */
function updateUIState(data) {
    appState.training = data.active;
    appState.sessionId = data.session_id;
    
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const progressSection = document.getElementById('progressSection');
    const metricsSection = document.getElementById('metricsSection');
    const currentConfig = document.getElementById('currentConfig');
    
    if (data.active) {
        // Entraînement en cours
        startBtn.disabled = true;
        stopBtn.disabled = false;
        progressSection.style.display = 'block';
        metricsSection.style.display = 'block';
        
        if (data.config) {
            currentConfig.style.display = 'block';
            updateConfigDisplay(data.config);
        }
        
        if (data.progress) {
            updateTrainingProgress(data.progress);
        }
    } else {
        // Entraînement arrêté
        startBtn.disabled = false;
        stopBtn.disabled = true;
        progressSection.style.display = 'none';
        metricsSection.style.display = 'none';
        currentConfig.style.display = 'none';
    }
}

/**
 * Mise à jour de l'affichage de la configuration
 */
function updateConfigDisplay(config) {
    const container = document.getElementById('configDetails');
    
    const html = `
        <div class="row">
            <div class="col-md-6">
                <small><strong>Modèle:</strong> ${config.model_name || 'N/A'}</small><br>
                <small><strong>Époques:</strong> ${config.epochs || 'N/A'}</small>
            </div>
            <div class="col-md-6">
                <small><strong>Batch Size:</strong> ${config.batch_size || 'N/A'}</small><br>
                <small><strong>Learning Rate:</strong> ${config.learning_rate || 'N/A'}</small>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

/**
 * Mise à jour de la progression d'entraînement
 */
function updateTrainingProgress(progress) {
    // Mise à jour des éléments de progression
    document.getElementById('currentEpoch').textContent = progress.current_epoch || 0;
    document.getElementById('totalEpochs').textContent = progress.total_epochs || 0;
    document.getElementById('progressPercent').textContent = `${(progress.progress_percent || 0).toFixed(1)}%`;
    document.getElementById('eta').textContent = `ETA: ${progress.eta || '--:--:--'}`;
    
    // Mise à jour de la barre de progression
    const progressBar = document.getElementById('epochProgress');
    progressBar.style.width = `${progress.progress_percent || 0}%`;
    
    // Mise à jour des métriques
    document.getElementById('trainLoss').textContent = (progress.train_loss || 0).toFixed(4);
    document.getElementById('valLoss').textContent = (progress.val_loss || 0).toFixed(4);
    document.getElementById('trainAccuracy').textContent = `${(progress.train_accuracy || 0).toFixed(1)}%`;
    document.getElementById('valAccuracy').textContent = `${(progress.val_accuracy || 0).toFixed(1)}%`;
    
    // Mise à jour du graphique
    updateChart(progress);
}

/**
 * Ajout d'une entrée de log
 */
function addLogEntry(logEntry) {
    const container = document.getElementById('logContainer');
    
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    
    const timestamp = new Date(logEntry.timestamp).toLocaleTimeString();
    const levelClass = `log-${logEntry.level}`;
    
    entry.innerHTML = `
        <span class="log-timestamp">[${timestamp}]</span>
        <span class="${levelClass}">${logEntry.message}</span>
    `;
    
    container.appendChild(entry);
    
    // Garder seulement les derniers logs
    while (container.children.length > CONFIG.maxLogEntries) {
        container.removeChild(container.firstChild);
    }
    
    // Scroll vers le bas
    container.scrollTop = container.scrollHeight;
}

/**
 * Affichage du modal de configuration rapide
 */
function showConfigModal() {
    const modal = new bootstrap.Modal(document.getElementById('quickStartModal'));
    modal.show();
}

/**
 * Démarrage de l'entraînement
 */
async function startTraining() {
    try {
        const configSelect = document.getElementById('configSelect');
        const epochsInput = document.getElementById('epochsInput');
        const batchSizeSelect = document.getElementById('batchSizeSelect');
        const learningRateInput = document.getElementById('learningRateInput');
        const trainCsvInput = document.getElementById('trainCsvInput');
        const valCsvInput = document.getElementById('valCsvInput');
        
        // Validation
        if (!configSelect.value) {
            alert('Veuillez sélectionner une configuration de modèle');
            return;
        }
        
        if (!trainCsvInput.value || !valCsvInput.value) {
            alert('Veuillez spécifier les chemins vers les fichiers CSV');
            return;
        }
        
        // Préparation des données
        const requestData = {
            config_name: configSelect.value,
            epochs: parseInt(epochsInput.value),
            batch_size: parseInt(batchSizeSelect.value),
            learning_rate: parseFloat(learningRateInput.value),
            train_csv: trainCsvInput.value,
            val_csv: valCsvInput.value
        };
        
        console.log('🚀 Démarrage de l\'entraînement:', requestData);
        
        const response = await fetch('/api/start_training', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            // Fermer le modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('quickStartModal'));
            modal.hide();
            
            // Réinitialiser l'historique
            trainingHistory = {
                epochs: [],
                trainLoss: [],
                valLoss: [],
                trainAccuracy: [],
                valAccuracy: []
            };
            
            // Vider les logs précédents
            document.getElementById('logContainer').innerHTML = '';
            
            console.log('✅ Entraînement démarré:', result);
            addLogEntry({
                timestamp: new Date().toISOString(),
                message: `Entraînement démarré - Session ID: ${result.session_id}`,
                level: 'success'
            });
        } else {
            console.error('❌ Erreur de démarrage:', result);
            alert(`Erreur: ${result.error || 'Échec du démarrage de l\'entraînement'}`);
        }
        
    } catch (error) {
        console.error('❌ Erreur lors du démarrage:', error);
        alert('Erreur de communication avec le serveur');
    }
}

/**
 * Arrêt de l'entraînement
 */
async function stopTraining() {
    try {
        if (!confirm('Êtes-vous sûr de vouloir arrêter l\'entraînement en cours ?')) {
            return;
        }
        
        console.log('🛑 Arrêt de l\'entraînement demandé');
        
        const response = await fetch('/api/stop_training', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const result = await response.json();
        
        if (response.ok) {
            console.log('✅ Arrêt demandé:', result);
            addLogEntry({
                timestamp: new Date().toISOString(),
                message: 'Arrêt de l\'entraînement demandé par l\'utilisateur',
                level: 'warning'
            });
        } else {
            console.error('❌ Erreur d\'arrêt:', result);
            alert(`Erreur: ${result.error || 'Échec de l\'arrêt'}`);
        }
        
    } catch (error) {
        console.error('❌ Erreur lors de l\'arrêt:', error);
        alert('Erreur de communication avec le serveur');
    }
}

/**
 * Gestion de la fin d'entraînement
 */
function onTrainingComplete(data) {
    console.log('🎉 Entraînement terminé avec succès');
    
    addLogEntry({
        timestamp: new Date().toISOString(),
        message: '🎉 Entraînement terminé avec succès !',
        level: 'success'
    });
    
    // Mettre à jour l'interface
    updateUIState({ active: false });
    
    // Notification optionnelle
    if ('Notification' in window && Notification.permission === 'granted') {
        new Notification('NightScan Training', {
            body: 'L\'entraînement EfficientNet est terminé !',
            icon: '/static/icon.png'
        });
    }
}

/**
 * Utilitaires
 */

// Demander la permission pour les notifications
if ('Notification' in window && Notification.permission === 'default') {
    Notification.requestPermission();
}

// Gestion des erreurs globales
window.addEventListener('error', function(event) {
    console.error('❌ Erreur JavaScript:', event.error);
    addLogEntry({
        timestamp: new Date().toISOString(),
        message: `Erreur JavaScript: ${event.error.message}`,
        level: 'error'
    });
});

// Debug info
console.log('📋 Interface d\'entraînement EfficientNet initialisée');
console.log('🔧 Configuration:', CONFIG);