/**
 * NightScan Design System 2025 - Glassmorphism Theme
 * Modern, cohesive design language with glassmorphism effects
 */

import { Dimensions } from 'react-native';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

// Color Palette - Dark/Night theme optimized for wildlife monitoring
export const Colors = {
  // Primary Palette
  primary: {
    50: '#E8F5E8',
    100: '#C8E6C9', 
    200: '#A5D6A7',
    300: '#81C784',
    400: '#66BB6A',
    500: '#4CAF50', // Main brand color
    600: '#43A047',
    700: '#388E3C',
    800: '#2E7D32',
    900: '#1B5E20',
  },

  // Secondary Palette - Warm accents
  secondary: {
    50: '#FFF3E0',
    100: '#FFE0B2',
    200: '#FFCC80',
    300: '#FFB74D',
    400: '#FFA726',
    500: '#FF9800',
    600: '#FB8C00',
    700: '#F57C00',
    800: '#EF6C00',
    900: '#E65100',
  },

  // Neutral Palette - Glassmorphism base
  neutral: {
    0: '#FFFFFF',
    50: '#FAFAFA',
    100: '#F5F5F5',
    200: '#EEEEEE',
    300: '#E0E0E0',
    400: '#BDBDBD',
    500: '#9E9E9E',
    600: '#757575',
    700: '#616161',
    800: '#424242',
    900: '#212121',
    950: '#0A0A0A',
  },

  // Semantic Colors
  success: '#4CAF50',
  warning: '#FF9800',
  error: '#F44336',
  info: '#2196F3',

  // Glassmorphism Backgrounds
  glass: {
    light: 'rgba(255, 255, 255, 0.1)',
    medium: 'rgba(255, 255, 255, 0.15)',
    heavy: 'rgba(255, 255, 255, 0.25)',
    dark: 'rgba(0, 0, 0, 0.1)',
    darkMedium: 'rgba(0, 0, 0, 0.15)',
    darkHeavy: 'rgba(0, 0, 0, 0.25)',
  },

  // Gradient Overlays
  gradients: {
    primary: ['#4CAF50', '#66BB6A'],
    secondary: ['#FF9800', '#FFA726'],
    sunset: ['#FF5722', '#FF9800'],
    night: ['#1A1A2E', '#16213E'],
    glass: ['rgba(255, 255, 255, 0.2)', 'rgba(255, 255, 255, 0.05)'],
  },
};

// Typography Scale
export const Typography = {
  // Font Families
  fonts: {
    regular: 'System',
    medium: 'System',
    semiBold: 'System',
    bold: 'System',
    mono: 'Menlo',
  },

  // Font Weights
  weights: {
    regular: '400',
    medium: '500',
    semiBold: '600',
    bold: '700',
  },

  // Font Sizes - Responsive scale
  sizes: {
    xs: 12,
    sm: 14,
    base: 16,
    lg: 18,
    xl: 20,
    '2xl': 24,
    '3xl': 30,
    '4xl': 36,
    '5xl': 48,
    '6xl': 60,
  },

  // Line Heights
  lineHeights: {
    tight: 1.2,
    snug: 1.375,
    normal: 1.5,
    relaxed: 1.625,
    loose: 2,
  },

  // Letter Spacing
  letterSpacing: {
    tighter: -0.5,
    tight: -0.25,
    normal: 0,
    wide: 0.25,
    wider: 0.5,
    widest: 1,
  },
};

// Spacing Scale - 8pt grid system
export const Spacing = {
  0: 0,
  1: 4,
  2: 8,
  3: 12,
  4: 16,
  5: 20,
  6: 24,
  7: 28,
  8: 32,
  9: 36,
  10: 40,
  12: 48,
  16: 64,
  20: 80,
  24: 96,
  32: 128,
};

// Border Radius
export const BorderRadius = {
  none: 0,
  sm: 4,
  base: 8,
  md: 12,
  lg: 16,
  xl: 20,
  '2xl': 24,
  '3xl': 32,
  full: 9999,
};

// Shadows - Glassmorphism compatible
export const Shadows = {
  // iOS shadows
  sm: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  base: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  md: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
    elevation: 6,
  },
  lg: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.2,
    shadowRadius: 16,
    elevation: 12,
  },
  xl: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.25,
    shadowRadius: 24,
    elevation: 18,
  },

  // Glassmorphism specific shadows
  glass: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 12,
    elevation: 5,
  },
  glassHeavy: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.15,
    shadowRadius: 20,
    elevation: 10,
  },
};

// Layout Breakpoints
export const Breakpoints = {
  sm: 375,
  md: 768,
  lg: 1024,
  xl: 1280,
};

// Layout helpers
export const Layout = {
  window: {
    width: screenWidth,
    height: screenHeight,
  },
  isSmallDevice: screenWidth < Breakpoints.sm,
  isMediumDevice: screenWidth >= Breakpoints.sm && screenWidth < Breakpoints.md,
  isLargeDevice: screenWidth >= Breakpoints.md,
  
  // Safe areas and paddings
  horizontalPadding: Spacing[4],
  verticalPadding: Spacing[4],
  safeAreaTop: 44, // iPhone safe area
  tabBarHeight: 83,
};

// Animation Configuration
export const Animation = {
  // Duration presets
  duration: {
    fast: 200,
    normal: 300,
    slow: 500,
    slower: 700,
  },

  // Easing curves
  easing: {
    linear: 'linear',
    ease: 'ease',
    easeIn: 'ease-in',
    easeOut: 'ease-out',
    easeInOut: 'ease-in-out',
  },

  // Spring configs for React Native Animated
  spring: {
    gentle: {
      tension: 120,
      friction: 8,
    },
    medium: {
      tension: 150,
      friction: 10,
    },
    firm: {
      tension: 200,
      friction: 12,
    },
  },
};

// Glassmorphism Effects
export const GlassEffects = {
  // Blur configurations
  blur: {
    light: 10,
    medium: 20,
    heavy: 40,
  },

  // Border styles for glassmorphism
  border: {
    light: {
      borderWidth: 1,
      borderColor: 'rgba(255, 255, 255, 0.1)',
    },
    medium: {
      borderWidth: 1,
      borderColor: 'rgba(255, 255, 255, 0.2)',
    },
    heavy: {
      borderWidth: 1.5,
      borderColor: 'rgba(255, 255, 255, 0.3)',
    },
  },

  // Backdrop styles
  backdrop: {
    light: {
      backgroundColor: Colors.glass.light,
      backdropFilter: 'blur(10px)',
    },
    medium: {
      backgroundColor: Colors.glass.medium,
      backdropFilter: 'blur(20px)',
    },
    heavy: {
      backgroundColor: Colors.glass.heavy,
      backdropFilter: 'blur(40px)',
    },
  },
};

// Component Variants
export const ComponentVariants = {
  // Button variants
  button: {
    primary: {
      backgroundColor: Colors.primary[500],
      borderColor: Colors.primary[500],
    },
    secondary: {
      backgroundColor: 'transparent',
      borderColor: Colors.primary[500],
      borderWidth: 1,
    },
    glass: {
      backgroundColor: Colors.glass.medium,
      borderColor: 'rgba(255, 255, 255, 0.2)',
      borderWidth: 1,
    },
    ghost: {
      backgroundColor: 'transparent',
      borderColor: 'transparent',
    },
  },

  // Card variants
  card: {
    elevated: {
      backgroundColor: Colors.neutral[0],
      ...Shadows.base,
    },
    glass: {
      backgroundColor: Colors.glass.medium,
      ...GlassEffects.border.medium,
      ...Shadows.glass,
    },
    outlined: {
      backgroundColor: 'transparent',
      borderWidth: 1,
      borderColor: Colors.neutral[200],
    },
  },
};

// Theme object export
export const Theme = {
  colors: Colors,
  typography: Typography,
  spacing: Spacing,
  borderRadius: BorderRadius,
  shadows: Shadows,
  layout: Layout,
  animation: Animation,
  glassEffects: GlassEffects,
  componentVariants: ComponentVariants,
  breakpoints: Breakpoints,
};

export default Theme;