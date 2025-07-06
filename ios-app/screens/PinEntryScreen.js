import React, { useState, useRef } from 'react';
import {
  View,
  StyleSheet,
  ImageBackground,
  TextInput,
  Alert,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';

import GlassCard from '../components/ui/GlassCard';
import GlassButton from '../components/ui/GlassButton';
import Typography, { Heading, Body, Label } from '../components/ui/Typography';
import Theme from '../theme/DesignSystem';
import authService from '../services/auth';

export default function PinEntryScreen({ navigation }) {
  const [pin, setPin] = useState('');
  const [loading, setLoading] = useState(false);
  const [attempts, setAttempts] = useState(0);
  
  const pinInputRef = useRef(null);
  const maxAttempts = 3;

  const handlePinChange = (value) => {
    // Only allow digits and limit to 6 characters
    const numericValue = value.replace(/[^0-9]/g, '').slice(0, 6);
    setPin(numericValue);
  };

  const handleAuthenticate = async () => {
    if (pin.length < 4) {
      Alert.alert('PIN trop court', 'Le PIN doit contenir au moins 4 chiffres');
      return;
    }

    setLoading(true);
    
    try {
      const result = await authService.authenticateWithPIN(pin);
      
      if (result.success) {
        Alert.alert(
          'Accès autorisé',
          'Connexion réussie au système NightScan',
          [
            {
              text: 'Continuer',
              onPress: () => navigation.replace('Main'),
            },
          ]
        );
      } else {
        const newAttempts = attempts + 1;
        setAttempts(newAttempts);
        
        if (newAttempts >= maxAttempts) {
          Alert.alert(
            'Accès bloqué',
            'Trop de tentatives échouées. Veuillez contacter le propriétaire du Pi.',
            [
              {
                text: 'Fermer',
                onPress: () => navigation.goBack(),
              },
            ]
          );
        } else {
          Alert.alert(
            'PIN incorrect',
            `Il vous reste ${maxAttempts - newAttempts} tentative${maxAttempts - newAttempts > 1 ? 's' : ''}`,
            [
              {
                text: 'Réessayer',
                onPress: () => {
                  setPin('');
                  setTimeout(() => pinInputRef.current?.focus(), 100);
                },
              },
            ]
          );
        }
      }
    } catch (error) {
      Alert.alert('Erreur', 'Une erreur est survenue lors de l\'authentification');
      console.error('PIN authentication error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    Alert.alert(
      'Annuler la connexion',
      'Êtes-vous sûr de vouloir annuler ? Vous ne pourrez pas accéder au Pi sans autorisation.',
      [
        { text: 'Continuer', style: 'cancel' },
        { 
          text: 'Annuler', 
          style: 'destructive',
          onPress: () => navigation.goBack() 
        },
      ]
    );
  };

  const renderPinDots = (currentPin, maxLength = 6) => {
    return (
      <View style={styles.pinDots}>
        {Array.from({ length: maxLength }, (_, index) => (
          <View
            key={index}
            style={[
              styles.pinDot,
              index < currentPin.length && styles.pinDotFilled,
            ]}
          />
        ))}
      </View>
    );
  };

  const deviceMAC = authService.getDeviceMACAddress();

  return (
    <ImageBackground
      source={{ uri: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjQwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9ImEiIHgxPSIwJSIgeTE9IjAlIiB4Mj0iMTAwJSIgeTI9IjEwMCUiPjxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiMxODJBM0EiLz48c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiMwRjE0MjAiLz48L2xpbmVhckdyYWRpZW50PjwvZGVmcz48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ1cmwoI2EpIi8+PC9zdmc+' }}
      style={styles.backgroundImage}
      resizeMode="cover"
    >
      <LinearGradient
        colors={['rgba(15, 20, 32, 0.8)', 'rgba(24, 42, 58, 0.9)']}
        style={styles.gradientOverlay}
      >
        <KeyboardAvoidingView 
          style={styles.container}
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        >
          <View style={styles.content}>
            <GlassCard variant="medium" style={styles.authCard}>
              <View style={styles.header}>
                <Ionicons name="lock-closed" size={48} color={Theme.colors.warning} />
                <Heading level={3} color={Theme.colors.neutral[0]} style={styles.title}>
                  Accès sécurisé requis
                </Heading>
                <Body color={Theme.colors.neutral[300]} style={styles.subtitle}>
                  Ce Pi est protégé. Entrez le code PIN pour accéder.
                </Body>
              </View>

              <View style={styles.deviceInfo}>
                <Label color={Theme.colors.neutral[500]} style={styles.deviceLabel}>
                  Appareil non reconnu
                </Label>
                <Body color={Theme.colors.neutral[400]} style={styles.deviceMAC}>
                  MAC: {deviceMAC}
                </Body>
              </View>

              <View style={styles.pinSection}>
                <Label color={Theme.colors.neutral[400]} style={styles.pinLabel}>
                  Code PIN requis
                </Label>
                
                {renderPinDots(pin)}
                
                <TextInput
                  ref={pinInputRef}
                  style={styles.hiddenInput}
                  value={pin}
                  onChangeText={handlePinChange}
                  keyboardType="numeric"
                  maxLength={6}
                  secureTextEntry
                  autoFocus
                  blurOnSubmit={false}
                  onSubmitEditing={handleAuthenticate}
                />
              </View>

              {attempts > 0 && (
                <View style={styles.warningBox}>
                  <Ionicons name="warning" size={20} color={Theme.colors.error} />
                  <Body color={Theme.colors.error} style={styles.warningText}>
                    {attempts === 1 ? 'Première' : attempts === 2 ? 'Seconde' : 'Dernière'} tentative échouée
                  </Body>
                </View>
              )}

              <View style={styles.infoBox}>
                <Ionicons name="information-circle" size={20} color={Theme.colors.info} />
                <Body color={Theme.colors.neutral[400]} style={styles.infoText}>
                  Contactez le propriétaire du Pi si vous n'avez pas le code PIN
                </Body>
              </View>

              <View style={styles.actions}>
                <GlassButton
                  title="Annuler"
                  variant="ghost"
                  size="medium"
                  onPress={handleCancel}
                  style={styles.actionButton}
                />
                
                <GlassButton
                  title="Se connecter"
                  variant="primary"
                  size="medium"
                  onPress={handleAuthenticate}
                  disabled={pin.length < 4 || loading || attempts >= maxAttempts}
                  loading={loading}
                  style={styles.actionButton}
                />
              </View>
            </GlassCard>
          </View>
        </KeyboardAvoidingView>
      </LinearGradient>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  backgroundImage: {
    flex: 1,
  },
  gradientOverlay: {
    flex: 1,
  },
  container: {
    flex: 1,
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: Theme.spacing[4],
  },
  
  authCard: {
    paddingVertical: Theme.spacing[8],
  },
  header: {
    alignItems: 'center',
    marginBottom: Theme.spacing[6],
  },
  title: {
    marginTop: Theme.spacing[4],
    marginBottom: Theme.spacing[2],
    textAlign: 'center',
  },
  subtitle: {
    textAlign: 'center',
    paddingHorizontal: Theme.spacing[4],
  },
  
  deviceInfo: {
    alignItems: 'center',
    marginBottom: Theme.spacing[6],
    paddingVertical: Theme.spacing[3],
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: Theme.borderRadius.md,
  },
  deviceLabel: {
    marginBottom: Theme.spacing[1],
  },
  deviceMAC: {
    fontFamily: 'monospace',
  },
  
  pinSection: {
    alignItems: 'center',
    marginBottom: Theme.spacing[6],
  },
  pinLabel: {
    marginBottom: Theme.spacing[4],
    textAlign: 'center',
  },
  pinDots: {
    flexDirection: 'row',
    gap: Theme.spacing[3],
    marginBottom: Theme.spacing[4],
  },
  pinDot: {
    width: 16,
    height: 16,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: Theme.colors.neutral[500],
    backgroundColor: 'transparent',
  },
  pinDotFilled: {
    backgroundColor: Theme.colors.primary[500],
    borderColor: Theme.colors.primary[500],
  },
  hiddenInput: {
    position: 'absolute',
    opacity: 0,
    width: 1,
    height: 1,
  },
  
  warningBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    borderRadius: Theme.borderRadius.md,
    padding: Theme.spacing[3],
    marginBottom: Theme.spacing[4],
    gap: Theme.spacing[2],
  },
  warningText: {
    flex: 1,
  },
  
  infoBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: 'rgba(59, 130, 246, 0.1)',
    borderRadius: Theme.borderRadius.md,
    padding: Theme.spacing[3],
    marginBottom: Theme.spacing[6],
    gap: Theme.spacing[2],
  },
  infoText: {
    flex: 1,
    lineHeight: 20,
  },
  
  actions: {
    flexDirection: 'row',
    gap: Theme.spacing[3],
  },
  actionButton: {
    flex: 1,
  },
});