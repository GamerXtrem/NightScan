
# Endpoints API pour le nouveau système de nommage
from filename_utils import FilenameParser, FilenameGenerator

@app.route('/api/filename/parse', methods=['POST'])
def api_parse_filename():
    """Parse un nom de fichier et retourne les métadonnées"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'Filename required'}), 400
        
        parser = FilenameParser()
        parsed = parser.parse_filename(filename)
        
        # Convertir datetime en string pour JSON
        if parsed['timestamp']:
            parsed['timestamp'] = parsed['timestamp'].isoformat()
        
        return jsonify({
            'success': True,
            'parsed': parsed
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/filename/generate', methods=['POST'])
def api_generate_filename():
    """Génère un nouveau nom de fichier"""
    try:
        data = request.get_json()
        
        file_type = data.get('type', 'audio')
        extension = data.get('extension', 'wav')
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        
        generator = FilenameGenerator()
        
        if file_type == 'audio':
            filename = generator.generate_audio_filename(
                latitude=latitude,
                longitude=longitude
            )
        elif file_type == 'image':
            filename = generator.generate_image_filename(
                latitude=latitude,
                longitude=longitude
            )
        else:
            filename = generator.generate_filename(
                file_type, extension,
                latitude=latitude,
                longitude=longitude
            )
        
        return jsonify({
            'success': True,
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/files/statistics', methods=['GET'])
def api_files_statistics():
    """Statistiques sur les formats de fichiers"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Statistiques par format
        cursor.execute("""
            SELECT filename_format, COUNT(*) as count
            FROM detections
            GROUP BY filename_format
            ORDER BY count DESC
        """)
        format_stats = [{'format': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # Statistiques par type
        cursor.execute("""
            SELECT file_type, COUNT(*) as count
            FROM detections
            GROUP BY file_type
            ORDER BY count DESC
        """)
        type_stats = [{'type': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # Fichiers avec GPS
        cursor.execute("""
            SELECT COUNT(*) as total, 
                   SUM(gps_embedded) as with_gps
            FROM detections
        """)
        gps_stats = cursor.fetchone()
        
        conn.close()
        
        return jsonify({
            'success': True,
            'statistics': {
                'formats': format_stats,
                'types': type_stats,
                'gps': {
                    'total': gps_stats[0],
                    'with_gps': gps_stats[1],
                    'without_gps': gps_stats[0] - gps_stats[1]
                }
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
