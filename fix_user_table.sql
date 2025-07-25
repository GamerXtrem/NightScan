-- Script pour ajouter les colonnes manquantes à la table user

-- Ajouter la colonne email
ALTER TABLE "user" ADD COLUMN IF NOT EXISTS email VARCHAR(200);

-- Ajouter la colonne is_active
ALTER TABLE "user" ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;

-- Ajouter la colonne created_at
ALTER TABLE "user" ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

-- Ajouter la colonne subscription_plan
ALTER TABLE "user" ADD COLUMN IF NOT EXISTS subscription_plan VARCHAR(50) DEFAULT 'free';

-- Mettre à jour l'utilisateur test existant
UPDATE "user" 
SET email = 'testuser@example.com',
    is_active = TRUE,
    created_at = CURRENT_TIMESTAMP,
    subscription_plan = 'free'
WHERE username = 'testuser';