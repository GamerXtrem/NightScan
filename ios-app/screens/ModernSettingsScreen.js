import React, { useState, useEffect } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  Alert,
  ImageBackground,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';

import GlassCard from '../components/ui/GlassCard';
import GlassButton from '../components/ui/GlassButton';
import Typography, { Heading, Body, Label } from '../components/ui/Typography';
import Theme from '../theme/DesignSystem';
import authService from '../services/auth';
import ChangePasswordModal from '../components/ChangePasswordModal';

export default function ModernSettingsScreen({ navigation }) {
  const [userInfo, setUserInfo] = useState(null);
  const [appVersion] = useState('1.0.0');
  const [showChangePassword, setShowChangePassword] = useState(false);

  useEffect(() => {
    loadUserInfo();
  }, []);

  const loadUserInfo = async () => {
    // Mock data - replace with actual user info
    setUserInfo({
      username: 'Utilisateur',
      email: 'user@nightscan.fr',
      isLoggedIn: false,
    });
  };

  const handleLogout = () => {
    Alert.alert(
      'Déconnexion',
      'Êtes-vous sûr de vouloir vous déconnecter ?',
      [
        { text: 'Annuler', style: 'cancel' },
        { text: 'Déconnexion', style: 'destructive', onPress: () => {
          // Handle logout
          navigation.navigate('Login');
        }},
      ]
    );
  };

  const handleAddUser = () => {
    Alert.prompt(
      'Ajouter Utilisateur',
      'Entrez l\'adresse MAC de l\'appareil à autoriser (format: XX:XX:XX:XX:XX:XX)',
      [
        { text: 'Annuler', style: 'cancel' },
        { 
          text: 'Ajouter', 
          onPress: async (macAddress) => {
            if (!macAddress || !isValidMAC(macAddress)) {
              Alert.alert('Erreur', 'Format d\'adresse MAC invalide');
              return;
            }
            
            const result = await authService.addUserByMAC(macAddress.trim());
            Alert.alert(
              result.success ? 'Succès' : 'Erreur',
              result.message
            );
          }
        },
      ],
      'plain-text',
      '',
      'default'
    );
  };

  const handleResetPi = () => {
    Alert.prompt(
      'Réinitialiser Pi',
      'Cette action supprimera tous les utilisateurs autorisés. Entrez votre PIN pour confirmer :',
      [
        { text: 'Annuler', style: 'cancel' },
        { 
          text: 'Réinitialiser', 
          style: 'destructive',
          onPress: async (pin) => {
            if (!pin) {
              Alert.alert('Erreur', 'PIN requis');
              return;
            }
            
            const result = await authService.resetPi(pin);
            if (result.success) {
              Alert.alert(
                'Pi Réinitialisé',
                'Le Pi a été réinitialisé avec succès. L\'application va se fermer.',
                [
                  { 
                    text: 'OK', 
                    onPress: () => {
                      // Clear auth and restart app
                      authService.clearAuthState();
                      navigation.reset({
                        index: 0,
                        routes: [{ name: 'Main' }],
                      });
                    }
                  }
                ]
              );
            } else {
              Alert.alert('Erreur', result.message);
            }
          }
        },
      ],
      'secure-text',
      '',
      'default'
    );
  };

  const isValidMAC = (mac) => {
    const macRegex = /^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$/;
    return macRegex.test(mac);
  };

  const renderHeader = () => (
    <View style={styles.header}>
      <Heading level={1} color={Theme.colors.neutral[0]} weight="bold">
        Paramètres
      </Heading>
      <Body color={Theme.colors.neutral[200]} style={styles.subtitle}>
        Configuration et gestion
      </Body>
    </View>
  );

  const renderUserSection = () => (
    <GlassCard variant="medium" style={styles.section}>
      <View style={styles.sectionHeader}>
        <Ionicons name="person-circle" size={24} color={Theme.colors.primary[500]} />
        <Heading level={5} style={styles.sectionTitle}>
          Compte Utilisateur
        </Heading>
      </View>

      {userInfo?.isLoggedIn ? (
        <View style={styles.userInfo}>
          <View style={styles.userDetails}>
            <Body weight="medium">{userInfo.username}</Body>
            <Label size="small" color={Theme.colors.neutral[600]}>
              {userInfo.email}
            </Label>
          </View>
          
          <View style={styles.accountActions}>
            <GlassButton
              title="Changer mot de passe"
              variant="glass"
              size="small"
              onPress={() => setShowChangePassword(true)}
              style={styles.changePasswordButton}
              icon="key"
            />
            
            <GlassButton
              title="Déconnexion"
              variant="danger"
              size="medium"
              onPress={handleLogout}
              style={styles.logoutButton}
            />
          </View>
        </View>
      ) : (
        <View style={styles.authButtons}>
          <GlassButton
            title="Connexion"
            variant="primary"
            size="medium"
            onPress={() => navigation.navigate('Login')}
            style={styles.authButton}
          />
          
          <GlassButton
            title="Inscription"
            variant="secondary"
            size="medium"
            onPress={() => navigation.navigate('Register')}
            style={styles.authButton}
          />
        </View>
      )}
    </GlassCard>
  );

  const renderHardwareSection = () => (
    <GlassCard variant="medium" style={styles.section}>
      <View style={styles.sectionHeader}>
        <Ionicons name="hardware-chip" size={24} color={Theme.colors.secondary[500]} />
        <Heading level={5} style={styles.sectionTitle}>
          Configuration Matériel
        </Heading>
      </View>

      <View style={styles.settingsList}>
        <GlassButton
          title="Prévisualisation Caméra"
          icon="camera"
          variant="glass"
          size="medium"
          onPress={() => navigation.navigate('CameraPreview')}
          style={styles.settingButton}
        />
        
        <GlassButton
          title="Installation Pi"
          icon="construct"
          variant="glass"
          size="medium"
          onPress={() => navigation.navigate('PiInstallation')}
          style={styles.settingButton}
        />
        
        <GlassButton
          title="Sensibilité Microphone"
          icon="mic"
          variant="glass"
          size="medium"
          onPress={() => navigation.navigate('AudioThreshold')}
          style={styles.settingButton}
        />
        
        <GlassButton
          title="Gestion Énergétique"
          icon="battery-charging"
          variant="glass"
          size="medium"
          onPress={() => navigation.navigate('EnergyManagement')}
          style={styles.settingButton}
        />
      </View>
    </GlassCard>
  );

  const renderSecuritySection = () => (
    <GlassCard variant="medium" style={styles.section}>
      <View style={styles.sectionHeader}>
        <Ionicons name="shield-checkmark" size={24} color={Theme.colors.warning} />
        <Heading level={5} style={styles.sectionTitle}>
          Sécurité et Accès
        </Heading>
      </View>

      <View style={styles.settingsList}>
        <GlassButton
          title="Ajouter Utilisateur"
          icon="person-add"
          variant="glass"
          size="medium"
          onPress={() => handleAddUser()}
          style={styles.settingButton}
        />
        
        <GlassButton
          title="Gérer Utilisateurs"
          icon="people"
          variant="glass"
          size="medium"
          onPress={() => navigation.navigate('UserManagement')}
          style={styles.settingButton}
        />
        
        <GlassButton
          title="Réinitialiser Pi"
          icon="refresh-circle"
          variant="glass"
          size="medium"
          onPress={() => handleResetPi()}
          style={styles.settingButton}
        />
      </View>
    </GlassCard>
  );

  const renderDataSection = () => (
    <GlassCard variant="medium" style={styles.section}>
      <View style={styles.sectionHeader}>
        <Ionicons name="server" size={24} color={Theme.colors.info} />
        <Heading level={5} style={styles.sectionTitle}>
          Données et Synchronisation
        </Heading>
      </View>

      <View style={styles.settingsList}>
        <GlassButton
          title="Synchroniser Maintenant"
          icon="sync"
          variant="glass"
          size="medium"
          onPress={() => {
            Alert.alert('Synchronisation', 'Synchronisation en cours...');
          }}
          style={styles.settingButton}
        />
        
        <GlassButton
          title="Effacer Cache"
          icon="trash"
          variant="glass"
          size="medium"
          onPress={() => {
            Alert.alert(
              'Effacer Cache',
              'Êtes-vous sûr de vouloir effacer le cache ?',
              [
                { text: 'Annuler', style: 'cancel' },
                { text: 'Effacer', style: 'destructive' },
              ]
            );
          }}
          style={styles.settingButton}
        />
        
        <GlassButton
          title="Exporter Données"
          icon="download"
          variant="glass"
          size="medium"
          onPress={() => {
            Alert.alert('Export', 'Fonctionnalité bientôt disponible');
          }}
          style={styles.settingButton}
        />
      </View>
    </GlassCard>
  );

  const renderAppSection = () => (
    <GlassCard variant="medium" style={styles.section}>
      <View style={styles.sectionHeader}>
        <Ionicons name="phone-portrait" size={24} color={Theme.colors.warning} />
        <Heading level={5} style={styles.sectionTitle}>
          Application
        </Heading>
      </View>

      <View style={styles.settingsList}>
        <GlassButton
          title="Mode Sombre"
          icon="moon"
          variant="glass"
          size="medium"
          onPress={() => {
            Alert.alert('Mode Sombre', 'Fonctionnalité bientôt disponible');
          }}
          style={styles.settingButton}
        />
        
        <GlassButton
          title="Notifications"
          icon="notifications"
          variant="glass"
          size="medium"
          onPress={() => {
            Alert.alert('Notifications', 'Configuration des notifications');
          }}
          style={styles.settingButton}
        />
        
        <GlassButton
          title="À Propos"
          icon="information-circle"
          variant="glass"
          size="medium"
          onPress={() => {
            Alert.alert('NightScan', `Version ${appVersion}\n\nSurveillance nocturne intelligente`);
          }}
          style={styles.settingButton}
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
        <ScrollView
          style={styles.container}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {renderHeader()}
          {renderUserSection()}
          {renderHardwareSection()}
          {renderSecuritySection()}
          {renderDataSection()}
          {renderAppSection()}
        </ScrollView>
        
        <ChangePasswordModal
          visible={showChangePassword}
          onClose={() => setShowChangePassword(false)}
          onSuccess={() => {
            Alert.alert('Succès', 'Votre mot de passe a été modifié avec succès.');
          }}
        />
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
  scrollContent: {
    paddingTop: Theme.layout.safeAreaTop + Theme.spacing[4],
    paddingHorizontal: Theme.spacing[4],
    paddingBottom: Theme.spacing[8],
  },
  header: {
    marginBottom: Theme.spacing[6],
  },
  subtitle: {
    marginTop: Theme.spacing[1],
  },
  
  section: {
    marginBottom: Theme.spacing[4],
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: Theme.spacing[4],
  },
  sectionTitle: {
    marginLeft: Theme.spacing[2],
  },
  
  // User Section
  userInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  userDetails: {
    flex: 1,
  },
  accountActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Theme.spacing[2],
  },
  changePasswordButton: {
    paddingHorizontal: Theme.spacing[3],
    paddingVertical: Theme.spacing[2],
  },
  logoutButton: {
    marginLeft: Theme.spacing[2],
  },
  authButtons: {
    flexDirection: 'row',
    gap: Theme.spacing[2],
  },
  authButton: {
    flex: 1,
  },
  
  // Settings Lists
  settingsList: {
    gap: Theme.spacing[2],
  },
  settingButton: {
    justifyContent: 'flex-start',
    paddingHorizontal: Theme.spacing[4],
  },
});