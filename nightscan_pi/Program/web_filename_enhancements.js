
// Fonctions JavaScript pour le nouveau système de nommage

// Parse un nom de fichier
async function parseFilename(filename) {
    try {
        const response = await fetch('/api/filename/parse', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({filename: filename})
        });
        
        const data = await response.json();
        if (data.success) {
            return data.parsed;
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        console.error('Erreur parsing filename:', error);
        return null;
    }
}

// Génère un nouveau nom de fichier
async function generateFilename(type, extension, latitude, longitude) {
    try {
        const response = await fetch('/api/filename/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                type: type,
                extension: extension,
                latitude: latitude,
                longitude: longitude
            })
        });
        
        const data = await response.json();
        if (data.success) {
            return data.filename;
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        console.error('Erreur génération filename:', error);
        return null;
    }
}

// Affiche les informations d'un fichier
function displayFileInfo(filename) {
    parseFilename(filename).then(parsed => {
        if (parsed) {
            const infoDiv = document.getElementById('file-info');
            if (infoDiv) {
                infoDiv.innerHTML = `
                    <div class="file-info">
                        <h4>Informations du fichier: ${filename}</h4>
                        <p><strong>Type:</strong> ${parsed.type}</p>
                        <p><strong>Format:</strong> ${parsed.format}</p>
                        <p><strong>Timestamp:</strong> ${parsed.timestamp || 'N/A'}</p>
                        <p><strong>GPS:</strong> ${parsed.latitude && parsed.longitude ? 
                            `${parsed.latitude.toFixed(4)}, ${parsed.longitude.toFixed(4)}` : 'N/A'}</p>
                        <p><strong>Zone:</strong> ${parsed.zone || 'N/A'}</p>
                    </div>
                `;
            }
        }
    });
}

// Charge les statistiques des fichiers
async function loadFileStatistics() {
    try {
        const response = await fetch('/api/files/statistics');
        const data = await response.json();
        
        if (data.success) {
            const statsDiv = document.getElementById('file-statistics');
            if (statsDiv) {
                const stats = data.statistics;
                
                let html = '<div class="file-statistics">';
                html += '<h3>Statistiques des fichiers</h3>';
                
                // Formats
                html += '<div class="stat-section">';
                html += '<h4>Par format:</h4>';
                html += '<ul>';
                stats.formats.forEach(format => {
                    html += `<li>${format.format}: ${format.count} fichiers</li>`;
                });
                html += '</ul>';
                html += '</div>';
                
                // Types
                html += '<div class="stat-section">';
                html += '<h4>Par type:</h4>';
                html += '<ul>';
                stats.types.forEach(type => {
                    html += `<li>${type.type}: ${type.count} fichiers</li>`;
                });
                html += '</ul>';
                html += '</div>';
                
                // GPS
                html += '<div class="stat-section">';
                html += '<h4>Localisation GPS:</h4>';
                html += `<p>Total: ${stats.gps.total}</p>`;
                html += `<p>Avec GPS: ${stats.gps.with_gps}</p>`;
                html += `<p>Sans GPS: ${stats.gps.without_gps}</p>`;
                html += '</div>';
                
                html += '</div>';
                statsDiv.innerHTML = html;
            }
        }
    } catch (error) {
        console.error('Erreur chargement statistiques:', error);
    }
}

// Initialisation au chargement de la page
document.addEventListener('DOMContentLoaded', function() {
    loadFileStatistics();
    
    // Ajouter des écouteurs d'événements pour les noms de fichiers
    document.querySelectorAll('.filename').forEach(element => {
        element.addEventListener('click', function() {
            displayFileInfo(this.textContent);
        });
    });
});
