import React from 'react';
import { Text, StyleSheet } from 'react-native';
import Theme from '../../theme/DesignSystem';

const Typography = ({ 
  variant = 'body',
  weight = 'regular',
  color = Theme.colors.neutral[900],
  align = 'left',
  numberOfLines,
  style,
  children,
  ...props 
}) => {
  const variants = {
    // Display variants
    display: {
      fontSize: Theme.typography.sizes['6xl'],
      lineHeight: Theme.typography.sizes['6xl'] * Theme.typography.lineHeights.tight,
      fontWeight: Theme.typography.weights.bold,
    },
    h1: {
      fontSize: Theme.typography.sizes['5xl'],
      lineHeight: Theme.typography.sizes['5xl'] * Theme.typography.lineHeights.tight,
      fontWeight: Theme.typography.weights.bold,
    },
    h2: {
      fontSize: Theme.typography.sizes['4xl'],
      lineHeight: Theme.typography.sizes['4xl'] * Theme.typography.lineHeights.tight,
      fontWeight: Theme.typography.weights.semiBold,
    },
    h3: {
      fontSize: Theme.typography.sizes['3xl'],
      lineHeight: Theme.typography.sizes['3xl'] * Theme.typography.lineHeights.snug,
      fontWeight: Theme.typography.weights.semiBold,
    },
    h4: {
      fontSize: Theme.typography.sizes['2xl'],
      lineHeight: Theme.typography.sizes['2xl'] * Theme.typography.lineHeights.snug,
      fontWeight: Theme.typography.weights.medium,
    },
    h5: {
      fontSize: Theme.typography.sizes.xl,
      lineHeight: Theme.typography.sizes.xl * Theme.typography.lineHeights.snug,
      fontWeight: Theme.typography.weights.medium,
    },
    h6: {
      fontSize: Theme.typography.sizes.lg,
      lineHeight: Theme.typography.sizes.lg * Theme.typography.lineHeights.normal,
      fontWeight: Theme.typography.weights.medium,
    },

    // Body variants
    body: {
      fontSize: Theme.typography.sizes.base,
      lineHeight: Theme.typography.sizes.base * Theme.typography.lineHeights.normal,
      fontWeight: Theme.typography.weights.regular,
    },
    bodyLarge: {
      fontSize: Theme.typography.sizes.lg,
      lineHeight: Theme.typography.sizes.lg * Theme.typography.lineHeights.normal,
      fontWeight: Theme.typography.weights.regular,
    },
    bodySmall: {
      fontSize: Theme.typography.sizes.sm,
      lineHeight: Theme.typography.sizes.sm * Theme.typography.lineHeights.normal,
      fontWeight: Theme.typography.weights.regular,
    },

    // Label variants
    label: {
      fontSize: Theme.typography.sizes.sm,
      lineHeight: Theme.typography.sizes.sm * Theme.typography.lineHeights.snug,
      fontWeight: Theme.typography.weights.medium,
      letterSpacing: Theme.typography.letterSpacing.wide,
    },
    labelLarge: {
      fontSize: Theme.typography.sizes.base,
      lineHeight: Theme.typography.sizes.base * Theme.typography.lineHeights.snug,
      fontWeight: Theme.typography.weights.medium,
      letterSpacing: Theme.typography.letterSpacing.wide,
    },
    labelSmall: {
      fontSize: Theme.typography.sizes.xs,
      lineHeight: Theme.typography.sizes.xs * Theme.typography.lineHeights.snug,
      fontWeight: Theme.typography.weights.medium,
      letterSpacing: Theme.typography.letterSpacing.wider,
    },

    // Utility variants
    caption: {
      fontSize: Theme.typography.sizes.xs,
      lineHeight: Theme.typography.sizes.xs * Theme.typography.lineHeights.normal,
      fontWeight: Theme.typography.weights.regular,
    },
    overline: {
      fontSize: Theme.typography.sizes.xs,
      lineHeight: Theme.typography.sizes.xs * Theme.typography.lineHeights.normal,
      fontWeight: Theme.typography.weights.medium,
      letterSpacing: Theme.typography.letterSpacing.widest,
      textTransform: 'uppercase',
    },
    mono: {
      fontSize: Theme.typography.sizes.sm,
      lineHeight: Theme.typography.sizes.sm * Theme.typography.lineHeights.normal,
      fontFamily: Theme.typography.fonts.mono,
      fontWeight: Theme.typography.weights.regular,
    },
  };

  const textStyle = [
    variants[variant] || variants.body,
    {
      color,
      textAlign: align,
      fontWeight: Theme.typography.weights[weight] || weight,
    },
    style,
  ];

  return (
    <Text 
      style={textStyle}
      numberOfLines={numberOfLines}
      {...props}
    >
      {children}
    </Text>
  );
};

// Preset components for common use cases
export const Heading = ({ level = 1, ...props }) => (
  <Typography variant={`h${level}`} {...props} />
);

export const Body = (props) => (
  <Typography variant="body" {...props} />
);

export const Label = ({ size = 'base', ...props }) => (
  <Typography variant={size === 'small' ? 'labelSmall' : size === 'large' ? 'labelLarge' : 'label'} {...props} />
);

export const Caption = (props) => (
  <Typography variant="caption" {...props} />
);

export const Overline = (props) => (
  <Typography variant="overline" {...props} />
);

export const Mono = (props) => (
  <Typography variant="mono" {...props} />
);

export default Typography;