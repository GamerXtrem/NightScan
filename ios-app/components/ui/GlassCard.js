import React from 'react';
import { View, StyleSheet, TouchableOpacity } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import Theme from '../../theme/DesignSystem';

const GlassCard = ({ 
  children, 
  variant = 'medium',
  onPress,
  style,
  contentStyle,
  blurIntensity,
  borderRadius = Theme.borderRadius.lg,
  ...props 
}) => {
  const variants = {
    light: {
      backgroundColor: Theme.colors.glass.light,
      borderColor: 'rgba(255, 255, 255, 0.1)',
      blurIntensity: blurIntensity || 10,
    },
    medium: {
      backgroundColor: Theme.colors.glass.medium,
      borderColor: 'rgba(255, 255, 255, 0.2)',
      blurIntensity: blurIntensity || 20,
    },
    heavy: {
      backgroundColor: Theme.colors.glass.heavy,
      borderColor: 'rgba(255, 255, 255, 0.3)',
      blurIntensity: blurIntensity || 40,
    },
    dark: {
      backgroundColor: Theme.colors.glass.darkMedium,
      borderColor: 'rgba(255, 255, 255, 0.1)',
      blurIntensity: blurIntensity || 20,
    },
  };

  const currentVariant = variants[variant] || variants.medium;

  const cardStyle = [
    styles.container,
    {
      backgroundColor: currentVariant.backgroundColor,
      borderColor: currentVariant.borderColor,
      borderRadius,
      ...Theme.shadows.glass,
    },
    style,
  ];

  const content = (
    <View style={[styles.content, contentStyle]}>
      {children}
    </View>
  );

  if (onPress) {
    return (
      <TouchableOpacity 
        style={cardStyle}
        onPress={onPress}
        activeOpacity={0.8}
        {...props}
      >
        {content}
      </TouchableOpacity>
    );
  }

  return (
    <View style={cardStyle} {...props}>
      {content}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    borderWidth: 1,
    overflow: 'hidden',
  },
  content: {
    padding: Theme.spacing[4],
  },
});

export default GlassCard;