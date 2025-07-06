import React, { useEffect, useState } from 'react';
import { View, StyleSheet, ImageBackground } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';

import GlassCard from './ui/GlassCard';
import { Heading, Body } from './ui/Typography';
import Theme from '../theme/DesignSystem';
import authService from '../services/auth';

export default function AuthWrapper({ children, navigation }) {
  const [authStatus, setAuthStatus] = useState('checking'); // 'checking', 'needsPairing', 'needsPIN', 'authenticated'
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkAuthenticationStatus();
  }, []);

  const checkAuthenticationStatus = async () => {
    try {
      setLoading(true);
      
      // Check if already authenticated
      if (authService.isAuthenticated()) {
        setAuthStatus('authenticated');
        return;
      }

      // Check pairing status with Pi
      const pairingStatus = await authService.checkPairingStatus();
      
      if (pairingStatus.requiresPairing) {
        // First time connecting - go to PIN setup
        setAuthStatus('needsPairing');
      } else if (pairingStatus.requiresPIN) {
        // MAC not recognized - need PIN entry
        setAuthStatus('needsPIN');
      } else if (pairingStatus.isOwner) {
        // Owner MAC recognized - auto authenticate
        setAuthStatus('authenticated');
      } else {
        // Unknown state
        setAuthStatus('needsPIN');
      }
    } catch (error) {
      console.error('Auth check failed:', error);
      // If Pi is not reachable, assume we need pairing
      setAuthStatus('needsPairing');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!loading) {
      if (authStatus === 'needsPairing') {
        navigation.navigate('PinSetup');
      } else if (authStatus === 'needsPIN') {
        navigation.navigate('PinEntry');
      }
      // If authenticated, just render children
    }
  }, [authStatus, loading, navigation]);

  if (loading) {
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
          <View style={styles.loadingContainer}>
            <GlassCard variant="medium" style={styles.loadingCard}>
              <View style={styles.loadingContent}>
                <Ionicons name="shield-checkmark" size={48} color={Theme.colors.primary[500]} />
                <Heading level={4} color={Theme.colors.neutral[0]} style={styles.loadingTitle}>
                  Vérification sécurisée
                </Heading>
                <Body color={Theme.colors.neutral[300]} style={styles.loadingText}>
                  Connexion au système NightScan...
                </Body>
              </View>
            </GlassCard>
          </View>
        </LinearGradient>
      </ImageBackground>
    );
  }

  // If authenticated, render the children (main app)
  if (authStatus === 'authenticated') {
    return children;
  }

  // Otherwise, navigation will handle showing PIN setup/entry screens
  return null;
}

const styles = StyleSheet.create({
  backgroundImage: {
    flex: 1,
  },
  gradientOverlay: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: Theme.spacing[4],
  },
  loadingCard: {
    width: '100%',
    maxWidth: 300,
    paddingVertical: Theme.spacing[8],
  },
  loadingContent: {
    alignItems: 'center',
  },
  loadingTitle: {
    marginTop: Theme.spacing[4],
    marginBottom: Theme.spacing[2],
    textAlign: 'center',
  },
  loadingText: {
    textAlign: 'center',
  },
});