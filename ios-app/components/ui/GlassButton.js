import React from 'react';
import { TouchableOpacity, StyleSheet, ActivityIndicator, View } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import Typography from './Typography';
import Theme from '../../theme/DesignSystem';

const GlassButton = ({
  title,
  onPress,
  variant = 'primary',
  size = 'medium',
  icon,
  iconPosition = 'left',
  loading = false,
  disabled = false,
  style,
  textStyle,
  ...props
}) => {
  const variants = {
    primary: {
      backgroundColor: Theme.colors.primary[500],
      borderColor: Theme.colors.primary[500],
      textColor: Theme.colors.neutral[0],
    },
    secondary: {
      backgroundColor: 'transparent',
      borderColor: Theme.colors.primary[500],
      textColor: Theme.colors.primary[500],
    },
    glass: {
      backgroundColor: Theme.colors.glass.medium,
      borderColor: 'rgba(255, 255, 255, 0.3)',
      textColor: Theme.colors.neutral[900],
    },
    glassDark: {
      backgroundColor: Theme.colors.glass.darkMedium,
      borderColor: 'rgba(255, 255, 255, 0.2)',
      textColor: Theme.colors.neutral[0],
    },
    ghost: {
      backgroundColor: 'transparent',
      borderColor: 'transparent',
      textColor: Theme.colors.primary[500],
    },
    danger: {
      backgroundColor: Theme.colors.error,
      borderColor: Theme.colors.error,
      textColor: Theme.colors.neutral[0],
    },
  };

  const sizes = {
    small: {
      paddingVertical: Theme.spacing[2],
      paddingHorizontal: Theme.spacing[3],
      borderRadius: Theme.borderRadius.base,
      fontSize: Theme.typography.sizes.sm,
      iconSize: 16,
    },
    medium: {
      paddingVertical: Theme.spacing[3],
      paddingHorizontal: Theme.spacing[4],
      borderRadius: Theme.borderRadius.md,
      fontSize: Theme.typography.sizes.base,
      iconSize: 20,
    },
    large: {
      paddingVertical: Theme.spacing[4],
      paddingHorizontal: Theme.spacing[6],
      borderRadius: Theme.borderRadius.lg,
      fontSize: Theme.typography.sizes.lg,
      iconSize: 24,
    },
  };

  const currentVariant = variants[variant] || variants.primary;
  const currentSize = sizes[size] || sizes.medium;

  const isDisabled = disabled || loading;
  const opacity = isDisabled ? 0.6 : 1;

  const buttonStyle = [
    styles.button,
    {
      backgroundColor: currentVariant.backgroundColor,
      borderColor: currentVariant.borderColor,
      borderRadius: currentSize.borderRadius,
      paddingVertical: currentSize.paddingVertical,
      paddingHorizontal: currentSize.paddingHorizontal,
      opacity,
      borderWidth: variant === 'secondary' || variant === 'glass' || variant === 'glassDark' ? 1 : 0,
      ...Theme.shadows.base,
    },
    style,
  ];

  const textColor = currentVariant.textColor;

  const renderIcon = () => {
    if (loading) {
      return (
        <ActivityIndicator 
          size="small" 
          color={textColor}
          style={styles.loadingIcon}
        />
      );
    }

    if (icon) {
      return (
        <Ionicons
          name={icon}
          size={currentSize.iconSize}
          color={textColor}
          style={[
            iconPosition === 'right' ? styles.iconRight : styles.iconLeft,
            title ? {} : styles.iconOnly
          ]}
        />
      );
    }

    return null;
  };

  const renderContent = () => {
    const iconElement = renderIcon();
    
    if (!title) {
      return iconElement;
    }

    return (
      <View style={styles.content}>
        {iconPosition === 'left' && iconElement}
        <Typography
          variant="label"
          weight="medium"
          color={textColor}
          style={[
            { fontSize: currentSize.fontSize },
            textStyle
          ]}
        >
          {title}
        </Typography>
        {iconPosition === 'right' && iconElement}
      </View>
    );
  };

  return (
    <TouchableOpacity
      style={buttonStyle}
      onPress={onPress}
      disabled={isDisabled}
      activeOpacity={0.8}
      {...props}
    >
      {renderContent()}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 44, // iOS minimum touch target
  },
  content: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconLeft: {
    marginRight: Theme.spacing[2],
  },
  iconRight: {
    marginLeft: Theme.spacing[2],
  },
  iconOnly: {
    margin: 0,
  },
  loadingIcon: {
    marginRight: Theme.spacing[2],
  },
});

export default GlassButton;