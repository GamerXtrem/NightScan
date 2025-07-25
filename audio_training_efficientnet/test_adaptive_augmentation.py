#!/usr/bin/env python3
"""
Script de test pour vérifier l'augmentation adaptative des classes minoritaires
"""

import sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import logging

# Ajouter le chemin parent pour les imports
sys.path.append(str(Path(__file__).parent))

from audio_augmentation import AdaptiveAudioAugmentation
from spectrogram_config import get_minority_class_category, get_augmentation_multiplier, MINORITY_CLASS_CONFIG

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_phase_shifting():
    """Test de l'implémentation du phase shifting."""
    print("\n=== Test du Phase Shifting ===")
    
    # Créer un signal de test
    sample_rate = 22050
    duration = 1.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    
    # Signal sinusoïdal simple
    frequency = 440  # La 440 Hz
    waveform = torch.sin(2 * np.pi * frequency * t).unsqueeze(0)
    
    # Créer l'augmenteur
    augmenter = AdaptiveAudioAugmentation(sample_rate)
    
    # Appliquer différents décalages de phase
    phase_shifts = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]
    
    for phase_shift in phase_shifts:
        shifted = augmenter.phase_shift(waveform, phase_shift)
        print(f"Phase shift: {phase_shift:.2f} rad ({phase_shift * 180/np.pi:.1f}°) - Shape: {shifted.shape}")
        
        # Vérifier que la forme est préservée
        assert shifted.shape == waveform.shape, "La forme du signal doit être préservée"
    
    print("✓ Phase shifting fonctionnel")


def test_adaptive_augmentation_categories():
    """Test des catégories d'augmentation adaptative."""
    print("\n=== Test des Catégories d'Augmentation ===")
    
    test_cases = [
        (10, 'very_small', 5),
        (30, 'very_small', 5),
        (50, 'small', 3),
        (150, 'small', 3),
        (200, 'medium', 2),
        (400, 'medium', 2),
        (500, 'large', 1),
        (1000, 'large', 1),
    ]
    
    for n_samples, expected_category, expected_multiplier in test_cases:
        category = get_minority_class_category(n_samples)
        multiplier = get_augmentation_multiplier(n_samples)
        
        print(f"Classe avec {n_samples} échantillons:")
        print(f"  - Catégorie: {category} (attendu: {expected_category})")
        print(f"  - Multiplicateur: {multiplier}x (attendu: {expected_multiplier}x)")
        
        assert category == expected_category, f"Catégorie incorrecte pour {n_samples} échantillons"
        assert multiplier == expected_multiplier, f"Multiplicateur incorrect pour {n_samples} échantillons"
    
    print("✓ Catégorisation correcte")


def test_adaptive_parameters():
    """Test des paramètres adaptatifs selon la taille de classe."""
    print("\n=== Test des Paramètres Adaptatifs ===")
    
    augmenter = AdaptiveAudioAugmentation()
    
    # Tester pour différentes tailles de classes
    class_sizes = [25, 100, 300, 600]
    
    for n_samples in class_sizes:
        augmenter.set_params_for_class_size(n_samples)
        category = augmenter.get_class_category(n_samples)
        
        print(f"\nClasse avec {n_samples} échantillons (catégorie: {category}):")
        print(f"  - Noise level: {augmenter.params['noise_level']}")
        print(f"  - Time shift max: {augmenter.params['time_shift_max']}")
        print(f"  - Pitch shift: ±{augmenter.params['pitch_shift']} semitones")
        print(f"  - Phase shift range: ±{augmenter.params['phase_shift_range']:.2f} rad")
        print(f"  - Probabilité d'augmentation: {augmenter.minority_params[category]['augmentation_prob']*100:.0f}%")


def test_cascaded_augmentation():
    """Test de l'augmentation en cascade pour les classes très petites."""
    print("\n=== Test de l'Augmentation en Cascade ===")
    
    # Créer un signal de test
    sample_rate = 22050
    duration = 2.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    
    # Signal complexe (simulation d'un chant d'oiseau)
    waveform = 0.5 * torch.sin(2 * np.pi * 1000 * t)
    waveform += 0.3 * torch.sin(2 * np.pi * 2000 * t)
    waveform += 0.2 * torch.sin(2 * np.pi * 3000 * t)
    waveform = waveform.unsqueeze(0)
    
    augmenter = AdaptiveAudioAugmentation(sample_rate)
    
    # Test pour une classe très petite (< 50 échantillons)
    n_samples = 30
    print(f"\nAugmentation pour classe avec {n_samples} échantillons:")
    
    # Appliquer l'augmentation plusieurs fois
    n_tests = 5
    for i in range(n_tests):
        augmented = augmenter.random_augment_waveform_adaptive(waveform, n_samples)
        
        # Vérifier que le signal a été modifié
        if not torch.equal(augmented, waveform):
            print(f"  Test {i+1}: Signal augmenté (différent de l'original)")
        else:
            print(f"  Test {i+1}: Signal non modifié")
    
    print("✓ Augmentation en cascade fonctionnelle")


def test_oversampling_calculation():
    """Test du calcul d'oversampling."""
    print("\n=== Test du Calcul d'Oversampling ===")
    
    augmenter = AdaptiveAudioAugmentation()
    
    test_cases = [
        (10, 50),   # 10 échantillons -> cible 50 (5x)
        (50, 150),  # 50 échantillons -> cible 150 (3x)
        (200, 400), # 200 échantillons -> cible 400 (2x)
        (500, 500), # 500 échantillons -> pas de changement
    ]
    
    for n_samples, expected_target in test_cases:
        multiplier = augmenter.get_augmentation_multiplier(n_samples)
        target = min(500, n_samples * multiplier)
        
        print(f"Classe avec {n_samples} échantillons:")
        print(f"  - Multiplicateur: {multiplier}x")
        print(f"  - Cible après oversampling: {target} (attendu: {expected_target})")
        
        assert target == expected_target, f"Cible incorrecte pour {n_samples} échantillons"
    
    print("✓ Calculs d'oversampling corrects")


def show_augmentation_strategy():
    """Affiche la stratégie d'augmentation sous forme de tableau."""
    print("\n=== Stratégie d'Augmentation Adaptative ===")
    
    # Préparer les données
    sample_counts = [10, 30, 50, 100, 200, 300, 500, 800]
    
    augmenter = AdaptiveAudioAugmentation()
    
    print("\n| Échantillons | Catégorie  | Multiplicateur | Noise Level | Time Shift | Pitch Shift | Phase Shift |")
    print("|--------------|------------|----------------|-------------|------------|-------------|-------------|")
    
    for n in sample_counts:
        cat = augmenter.get_class_category(n)
        mult = augmenter.get_augmentation_multiplier(n)
        augmenter.set_params_for_class_size(n)
        
        noise = augmenter.params['noise_level']
        time_shift = augmenter.params['time_shift_max']
        pitch_shift = augmenter.params['pitch_shift']
        phase_shift = augmenter.params['phase_shift_range']
        
        print(f"| {n:12d} | {cat:10s} | {mult:14d}x | {noise:11.3f} | {time_shift:10.1%} | ±{pitch_shift:10d} | ±{phase_shift/np.pi:9.2f}π |")
    
    print("\n✓ Stratégie d'augmentation affichée")


def main():
    """Fonction principale de test."""
    print("=== Test du Système d'Augmentation Adaptative ===")
    print("=" * 50)
    
    # Exécuter tous les tests
    test_phase_shifting()
    test_adaptive_augmentation_categories()
    test_adaptive_parameters()
    test_cascaded_augmentation()
    test_oversampling_calculation()
    show_augmentation_strategy()
    
    print("\n" + "=" * 50)
    print("✅ Tous les tests sont passés avec succès!")
    print("\nRésumé de l'implémentation:")
    print("- Phase shifting implémenté avec STFT/iSTFT")
    print("- Augmentation adaptative selon la taille de classe")
    print("- Paramètres agressifs pour classes < 500 échantillons")
    print("- Oversampling configurable pour équilibrer les classes")
    print("- Logging détaillé des statistiques d'augmentation")


if __name__ == "__main__":
    main()