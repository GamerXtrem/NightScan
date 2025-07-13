import React, { useState } from 'react';
import { View, TextInput, Button, StyleSheet, Alert, TouchableOpacity, Text } from 'react-native';
import { login as loginRequest, requestPasswordReset } from '../services/api';

export default function LoginScreen({ navigation }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleLogin = async () => {
    try {
      await loginRequest(username, password);
      navigation.navigate('Main');
    } catch {
      Alert.alert('Login failed');
    }
  };

  const handleForgotPassword = () => {
    Alert.prompt(
      'Mot de passe oublié',
      'Entrez votre adresse email pour recevoir les instructions de réinitialisation :',
      [
        { text: 'Annuler', style: 'cancel' },
        {
          text: 'Envoyer',
          onPress: async (email) => {
            if (!email || !email.includes('@')) {
              Alert.alert('Erreur', 'Veuillez entrer une adresse email valide');
              return;
            }
            
            try {
              await requestPasswordReset(email);
              Alert.alert(
                'Email envoyé',
                'Si un compte existe avec cette adresse email, vous recevrez les instructions de réinitialisation.'
              );
            } catch (error) {
              Alert.alert('Erreur', 'Impossible d\'envoyer l\'email de réinitialisation');
            }
          }
        }
      ],
      'plain-text',
      '',
      'email-address'
    );
  };

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.input}
        placeholder="Username"
        value={username}
        onChangeText={setUsername}
        autoCapitalize="none"
      />
      <TextInput
        style={styles.input}
        placeholder="Password"
        value={password}
        onChangeText={setPassword}
        secureTextEntry
      />
      <Button title="Login" onPress={handleLogin} />
      
      <TouchableOpacity onPress={handleForgotPassword} style={styles.forgotPassword}>
        <Text style={styles.forgotPasswordText}>Mot de passe oublié ?</Text>
      </TouchableOpacity>
      
      <View style={styles.spacing} />
      <Button title="Register" onPress={() => navigation.navigate('Register')} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    padding: 16,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    padding: 8,
    marginBottom: 12,
    borderRadius: 4,
  },
  spacing: {
    height: 12,
  },
  forgotPassword: {
    marginTop: 15,
    alignItems: 'center',
  },
  forgotPasswordText: {
    color: '#4CAF50',
    fontSize: 14,
    textDecorationLine: 'underline',
  },
});
