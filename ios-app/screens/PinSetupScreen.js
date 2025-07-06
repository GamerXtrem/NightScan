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

export default function PinSetupScreen({ navigation }) {
  const [pin, setPin] = useState('');
  const [confirmPin, setConfirmPin] = useState('');
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(1); // 1: enter PIN, 2: confirm PIN
  
  const pinInputRef = useRef(null);
  const confirmPinInputRef = useRef(null);

  const handlePinChange = (value) => {
    // Only allow digits and limit to 6 characters
    const numericValue = value.replace(/[^0-9]/g, '').slice(0, 6);
    setPin(numericValue);
  };

  const handleConfirmPinChange = (value) => {
    const numericValue = value.replace(/[^0-9]/g, '').slice(0, 6);
    setConfirmPin(numericValue);
  };

  const handleContinue = () => {
    if (step === 1) {
      if (pin.length < 4) {
        Alert.alert('PIN trop court', 'Le PIN doit contenir au moins 4 chiffres');
        return;
      }
      setStep(2);
      setTimeout(() => confirmPinInputRef.current?.focus(), 100);
    } else {
      handleSetupPin();
    }
  };

  const handleSetupPin = async () => {
    if (pin !== confirmPin) {
      Alert.alert('Erreur', 'Les codes PIN ne correspondent pas');
      setStep(1);
      setConfirmPin('');
      setTimeout(() => pinInputRef.current?.focus(), 100);
      return;
    }

    if (pin.length < 4) {
      Alert.alert('PIN trop court', 'Le PIN doit contenir au moins 4 chiffres');
      return;
    }

    setLoading(true);
    
    try {
      const result = await authService.performPairing(pin);
      
      if (result.success) {
        Alert.alert(
          'Configuration réussie !',
          'Votre Pi est maintenant configuré et sécurisé avec votre PIN.',
          [
            {
              text: 'Continuer',
              onPress: () => navigation.replace('Main'),
            },
          ]
        );
      } else {
        Alert.alert('Erreur', result.message || 'Configuration échouée');
      }
    } catch (error) {
      Alert.alert('Erreur', 'Une erreur est survenue lors de la configuration');
      console.error('PIN setup error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleBack = () => {
    if (step === 2) {
      setStep(1);
      setConfirmPin('');
      setTimeout(() => pinInputRef.current?.focus(), 100);
    } else {
      navigation.goBack();
    }
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

  const renderStep1 = () => (
    <GlassCard variant="medium" style={styles.setupCard}>
      <View style={styles.header}>
        <Ionicons name="shield-checkmark" size={48} color={Theme.colors.primary[500]} />
        <Heading level={3} color={Theme.colors.neutral[0]} style={styles.title}>
          Configuration sécurisée
        </Heading>
        <Body color={Theme.colors.neutral[300]} style={styles.subtitle}>
          Définissez un code PIN pour sécuriser l'accès à votre Pi
        </Body>
      </View>

      <View style={styles.pinSection}>
        <Label color={Theme.colors.neutral[400]} style={styles.pinLabel}>
          Entrez votre code PIN (4-6 chiffres)
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
          onSubmitEditing={handleContinue}
        />
      </View>

      <View style={styles.infoBox}>
        <Ionicons name="information-circle" size={20} color={Theme.colors.info} />
        <Body color={Theme.colors.neutral[400]} style={styles.infoText}>
          Ce PIN sera requis pour tout nouvel appareil tentant de se connecter à votre Pi
        </Body>
      </View>

      <View style={styles.actions}>
        <GlassButton
          title="Retour"
          variant="ghost"
          size="medium"
          onPress={handleBack}
          style={styles.actionButton}
        />
        
        <GlassButton
          title="Continuer"
          variant="primary"
          size="medium"
          onPress={handleContinue}
          disabled={pin.length < 4}
          style={styles.actionButton}
        />
      </View>
    </GlassCard>
  );

  const renderStep2 = () => (
    <GlassCard variant="medium" style={styles.setupCard}>
      <View style={styles.header}>
        <Ionicons name="checkmark-circle" size={48} color={Theme.colors.success} />
        <Heading level={3} color={Theme.colors.neutral[0]} style={styles.title}>
          Confirmez votre PIN
        </Heading>
        <Body color={Theme.colors.neutral[300]} style={styles.subtitle}>
          Saisissez à nouveau votre code PIN pour confirmer
        </Body>
      </View>

      <View style={styles.pinSection}>
        <Label color={Theme.colors.neutral[400]} style={styles.pinLabel}>
          Confirmez votre code PIN
        </Label>
        
        {renderPinDots(confirmPin)}
        
        <TextInput
          ref={confirmPinInputRef}
          style={styles.hiddenInput}
          value={confirmPin}
          onChangeText={handleConfirmPinChange}
          keyboardType="numeric"
          maxLength={6}
          secureTextEntry
          autoFocus
          blurOnSubmit={false}
          onSubmitEditing={handleContinue}
        />
      </View>

      <View style={styles.actions}>
        <GlassButton
          title="Retour"
          variant="ghost"
          size="medium"
          onPress={handleBack}
          style={styles.actionButton}
        />
        
        <GlassButton
          title="Terminer"
          variant="primary"
          size="medium"
          onPress={handleContinue}
          disabled={confirmPin.length < 4 || loading}
          loading={loading}
          style={styles.actionButton}
        />
      </View>
    </GlassCard>
  );

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
            {step === 1 ? renderStep1() : renderStep2()}
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
  
  setupCard: {
    paddingVertical: Theme.spacing[8],
  },
  header: {
    alignItems: 'center',
    marginBottom: Theme.spacing[8],
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