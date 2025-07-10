-- Migration SQL pour le système de rétention des données par paliers de quota
-- Exécuter ce script pour ajouter les fonctionnalités de rétention

-- 1. Ajouter le champ created_at à la table prediction
ALTER TABLE prediction 
ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

-- Mettre à jour les enregistrements existants avec une date approximative
UPDATE prediction 
SET created_at = CURRENT_TIMESTAMP - (id * INTERVAL '1 hour')
WHERE created_at IS NULL;

-- Rendre le champ non nullable après la mise à jour
ALTER TABLE prediction 
ALTER COLUMN created_at SET NOT NULL;

-- Ajouter un index sur created_at pour les requêtes de nettoyage
CREATE INDEX IF NOT EXISTS idx_prediction_created_at ON prediction(created_at DESC);

-- 2. Ajouter le champ data_retention_days à la table plan_features
ALTER TABLE plan_features 
ADD COLUMN IF NOT EXISTS data_retention_days INTEGER DEFAULT 30;

-- Définir les périodes de rétention par palier
UPDATE plan_features SET data_retention_days = 7 WHERE plan_type = 'free';
UPDATE plan_features SET data_retention_days = 30 WHERE plan_type = 'premium';
UPDATE plan_features SET data_retention_days = 180 WHERE plan_type = 'enterprise';

-- 3. Créer la table data_retention_log pour l'audit
CREATE TABLE IF NOT EXISTS data_retention_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES "user"(id),
    plan_type VARCHAR(50),
    retention_days INTEGER NOT NULL,
    records_deleted INTEGER NOT NULL DEFAULT 0,
    total_size_deleted_bytes BIGINT NOT NULL DEFAULT 0,
    deletion_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    retention_policy_version VARCHAR(50) DEFAULT '1.0',
    admin_override BOOLEAN DEFAULT FALSE,
    admin_reason TEXT,
    metadata JSONB
);

-- Index pour les requêtes d'audit
CREATE INDEX IF NOT EXISTS idx_data_retention_log_user ON data_retention_log(user_id);
CREATE INDEX IF NOT EXISTS idx_data_retention_log_date ON data_retention_log(deletion_date DESC);

-- 4. Créer la table prediction_archive pour le soft delete
CREATE TABLE IF NOT EXISTS prediction_archive (
    id SERIAL PRIMARY KEY,
    original_prediction_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    filename VARCHAR(200),
    result TEXT,
    file_size INTEGER,
    created_at TIMESTAMP NOT NULL,
    archived_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    plan_type_at_archive VARCHAR(50),
    retention_days INTEGER,
    archived_by VARCHAR(50) DEFAULT 'system'
);

-- Index pour les requêtes d'archive
CREATE INDEX IF NOT EXISTS idx_prediction_archive_user ON prediction_archive(user_id);
CREATE INDEX IF NOT EXISTS idx_prediction_archive_date ON prediction_archive(archived_at DESC);
CREATE INDEX IF NOT EXISTS idx_prediction_archive_original ON prediction_archive(original_prediction_id);

-- 5. Fonction pour identifier les prédictions expirées
CREATE OR REPLACE FUNCTION get_expired_predictions(user_id_param INTEGER DEFAULT NULL)
RETURNS TABLE(
    prediction_id INTEGER,
    user_id INTEGER,
    plan_type VARCHAR(50),
    retention_days INTEGER,
    created_at TIMESTAMP,
    days_old INTEGER,
    should_delete BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p.id as prediction_id,
        p.user_id,
        up.plan_type,
        pf.data_retention_days as retention_days,
        p.created_at,
        EXTRACT(DAY FROM (CURRENT_TIMESTAMP - p.created_at))::INTEGER as days_old,
        (EXTRACT(DAY FROM (CURRENT_TIMESTAMP - p.created_at)) > pf.data_retention_days) as should_delete
    FROM prediction p
    JOIN user_plans up ON p.user_id = up.user_id
    JOIN plan_features pf ON up.plan_type = pf.plan_type
    WHERE 
        (user_id_param IS NULL OR p.user_id = user_id_param)
        AND up.status = 'active'
        AND pf.is_active = true
        AND EXTRACT(DAY FROM (CURRENT_TIMESTAMP - p.created_at)) > pf.data_retention_days
    ORDER BY p.created_at ASC;
END;
$$ LANGUAGE plpgsql;

-- 6. Fonction pour archiver et supprimer les prédictions expirées
CREATE OR REPLACE FUNCTION cleanup_expired_predictions(
    user_id_param INTEGER DEFAULT NULL,
    dry_run BOOLEAN DEFAULT TRUE
) RETURNS JSONB AS $$
DECLARE
    deleted_count INTEGER := 0;
    total_size_deleted BIGINT := 0;
    user_record RECORD;
    prediction_record RECORD;
    log_entry_id INTEGER;
BEGIN
    -- Parcourir les utilisateurs avec des prédictions expirées
    FOR user_record IN 
        SELECT DISTINCT 
            ep.user_id,
            ep.plan_type,
            ep.retention_days,
            COUNT(*) as expired_count,
            SUM(p.file_size) as total_size
        FROM get_expired_predictions(user_id_param) ep
        JOIN prediction p ON ep.prediction_id = p.id
        WHERE ep.should_delete = true
        GROUP BY ep.user_id, ep.plan_type, ep.retention_days
    LOOP
        IF NOT dry_run THEN
            -- Créer une entrée de log
            INSERT INTO data_retention_log (
                user_id, 
                plan_type, 
                retention_days,
                records_deleted,
                total_size_deleted_bytes,
                metadata
            ) VALUES (
                user_record.user_id,
                user_record.plan_type,
                user_record.retention_days,
                user_record.expired_count,
                COALESCE(user_record.total_size, 0),
                jsonb_build_object(
                    'cleanup_type', 'automatic',
                    'dry_run', false,
                    'timestamp', CURRENT_TIMESTAMP
                )
            ) RETURNING id INTO log_entry_id;

            -- Archiver puis supprimer les prédictions expirées pour cet utilisateur
            FOR prediction_record IN
                SELECT ep.prediction_id, p.*
                FROM get_expired_predictions(user_record.user_id) ep
                JOIN prediction p ON ep.prediction_id = p.id
                WHERE ep.should_delete = true
            LOOP
                -- Archiver
                INSERT INTO prediction_archive (
                    original_prediction_id,
                    user_id,
                    filename,
                    result,
                    file_size,
                    created_at,
                    plan_type_at_archive,
                    retention_days
                ) SELECT 
                    id,
                    user_id,
                    filename,
                    result,
                    file_size,
                    created_at,
                    user_record.plan_type,
                    user_record.retention_days
                FROM prediction 
                WHERE id = prediction_record.prediction_id;

                -- Supprimer
                DELETE FROM prediction WHERE id = prediction_record.prediction_id;
                
                deleted_count := deleted_count + 1;
                total_size_deleted := total_size_deleted + COALESCE(prediction_record.file_size, 0);
            END LOOP;
        ELSE
            -- Mode dry_run - juste compter
            deleted_count := deleted_count + user_record.expired_count;
            total_size_deleted := total_size_deleted + COALESCE(user_record.total_size, 0);
        END IF;
    END LOOP;

    RETURN jsonb_build_object(
        'success', true,
        'dry_run', dry_run,
        'deleted_count', deleted_count,
        'total_size_deleted_bytes', total_size_deleted,
        'total_size_deleted_mb', ROUND(total_size_deleted / 1024.0 / 1024.0, 2),
        'timestamp', CURRENT_TIMESTAMP
    );
END;
$$ LANGUAGE plpgsql;

-- 7. Fonction pour obtenir les statistiques de rétention par utilisateur
CREATE OR REPLACE FUNCTION get_user_retention_stats(user_id_param INTEGER)
RETURNS JSONB AS $$
DECLARE
    stats RECORD;
    result JSONB;
BEGIN
    SELECT 
        up.plan_type,
        pf.data_retention_days,
        COUNT(p.id) as total_predictions,
        COUNT(CASE WHEN EXTRACT(DAY FROM (CURRENT_TIMESTAMP - p.created_at)) > pf.data_retention_days THEN 1 END) as expired_predictions,
        COUNT(CASE WHEN EXTRACT(DAY FROM (CURRENT_TIMESTAMP - p.created_at)) > (pf.data_retention_days - 7) THEN 1 END) as expiring_soon,
        MIN(p.created_at) as oldest_prediction,
        MAX(p.created_at) as newest_prediction,
        SUM(p.file_size) as total_size_bytes
    INTO stats
    FROM user_plans up
    JOIN plan_features pf ON up.plan_type = pf.plan_type
    LEFT JOIN prediction p ON up.user_id = p.user_id
    WHERE up.user_id = user_id_param
        AND up.status = 'active'
        AND pf.is_active = true
    GROUP BY up.plan_type, pf.data_retention_days;

    IF stats IS NULL THEN
        RETURN jsonb_build_object(
            'error', 'User not found or no active plan'
        );
    END IF;

    RETURN jsonb_build_object(
        'user_id', user_id_param,
        'plan_type', stats.plan_type,
        'retention_days', stats.data_retention_days,
        'total_predictions', COALESCE(stats.total_predictions, 0),
        'expired_predictions', COALESCE(stats.expired_predictions, 0),
        'expiring_soon', COALESCE(stats.expiring_soon, 0),
        'oldest_prediction', stats.oldest_prediction,
        'newest_prediction', stats.newest_prediction,
        'total_size_bytes', COALESCE(stats.total_size_bytes, 0),
        'total_size_mb', ROUND(COALESCE(stats.total_size_bytes, 0) / 1024.0 / 1024.0, 2)
    );
END;
$$ LANGUAGE plpgsql;

-- 8. Vue pour faciliter les requêtes de rétention
CREATE OR REPLACE VIEW prediction_retention_view AS
SELECT 
    p.id,
    p.user_id,
    p.filename,
    p.file_size,
    p.created_at,
    up.plan_type,
    pf.data_retention_days,
    EXTRACT(DAY FROM (CURRENT_TIMESTAMP - p.created_at))::INTEGER as days_old,
    (EXTRACT(DAY FROM (CURRENT_TIMESTAMP - p.created_at)) > pf.data_retention_days) as is_expired,
    (EXTRACT(DAY FROM (CURRENT_TIMESTAMP - p.created_at)) > (pf.data_retention_days - 7)) as expires_soon,
    (pf.data_retention_days - EXTRACT(DAY FROM (CURRENT_TIMESTAMP - p.created_at)))::INTEGER as days_until_expiry
FROM prediction p
JOIN user_plans up ON p.user_id = up.user_id
JOIN plan_features pf ON up.plan_type = pf.plan_type
WHERE up.status = 'active' AND pf.is_active = true;

-- 9. Trigger pour mettre à jour automatiquement les logs de rétention
CREATE OR REPLACE FUNCTION update_retention_metadata()
RETURNS TRIGGER AS $$
BEGIN
    -- Mettre à jour les métadonnées lors de la suppression de prédictions
    IF TG_OP = 'DELETE' THEN
        -- Log automatique si c'est une suppression liée à la rétention
        INSERT INTO data_retention_log (
            user_id,
            plan_type,
            retention_days,
            records_deleted,
            total_size_deleted_bytes,
            metadata
        ) 
        SELECT 
            OLD.user_id,
            up.plan_type,
            pf.data_retention_days,
            1,
            COALESCE(OLD.file_size, 0),
            jsonb_build_object(
                'cleanup_type', 'trigger',
                'prediction_id', OLD.id,
                'timestamp', CURRENT_TIMESTAMP
            )
        FROM user_plans up
        JOIN plan_features pf ON up.plan_type = pf.plan_type
        WHERE up.user_id = OLD.user_id 
            AND up.status = 'active'
            AND pf.is_active = true
            AND EXTRACT(DAY FROM (CURRENT_TIMESTAMP - OLD.created_at)) > pf.data_retention_days;
        
        RETURN OLD;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Activer le trigger
DROP TRIGGER IF EXISTS prediction_retention_trigger ON prediction;
CREATE TRIGGER prediction_retention_trigger
    BEFORE DELETE ON prediction
    FOR EACH ROW
    EXECUTE FUNCTION update_retention_metadata();

-- 10. Fonction de test pour simuler le système
CREATE OR REPLACE FUNCTION test_retention_system()
RETURNS JSONB AS $$
DECLARE
    result JSONB;
    test_results JSONB[] := ARRAY[]::JSONB[];
BEGIN
    -- Test 1: Vérifier la configuration des paliers
    SELECT jsonb_agg(
        jsonb_build_object(
            'plan_type', plan_type,
            'plan_name', plan_name,
            'retention_days', data_retention_days,
            'monthly_quota', monthly_quota
        )
    ) INTO result
    FROM plan_features WHERE is_active = true;
    
    test_results := test_results || jsonb_build_object('plans_config', result);
    
    -- Test 2: Statistiques globales
    SELECT jsonb_build_object(
        'total_users_with_plans', COUNT(DISTINCT up.user_id),
        'total_predictions', COUNT(p.id),
        'total_expired', COUNT(CASE WHEN prv.is_expired THEN 1 END),
        'total_expiring_soon', COUNT(CASE WHEN prv.expires_soon AND NOT prv.is_expired THEN 1 END)
    ) INTO result
    FROM user_plans up
    LEFT JOIN prediction p ON up.user_id = p.user_id
    LEFT JOIN prediction_retention_view prv ON p.id = prv.id
    WHERE up.status = 'active';
    
    test_results := test_results || jsonb_build_object('global_stats', result);
    
    -- Test 3: Dry run du nettoyage
    SELECT cleanup_expired_predictions(NULL, true) INTO result;
    test_results := test_results || jsonb_build_object('cleanup_dry_run', result);
    
    RETURN jsonb_build_object(
        'test_timestamp', CURRENT_TIMESTAMP,
        'system_status', 'ready',
        'tests', test_results
    );
END;
$$ LANGUAGE plpgsql;

-- Notification des changements
DO $$
BEGIN
    RAISE NOTICE '✅ Migration du système de rétention des données terminée avec succès!';
    RAISE NOTICE '📊 Paliers de rétention configurés:';
    RAISE NOTICE '   - Free: 7 jours';
    RAISE NOTICE '   - Premium: 30 jours'; 
    RAISE NOTICE '   - Enterprise: 180 jours (6 mois)';
    RAISE NOTICE '🔧 Fonctions disponibles:';
    RAISE NOTICE '   - get_expired_predictions(user_id)';
    RAISE NOTICE '   - cleanup_expired_predictions(user_id, dry_run)';
    RAISE NOTICE '   - get_user_retention_stats(user_id)';
    RAISE NOTICE '   - test_retention_system()';
    RAISE NOTICE '📈 Vue disponible: prediction_retention_view';
    RAISE NOTICE '🗂️ Tables créées: data_retention_log, prediction_archive';
END $$;