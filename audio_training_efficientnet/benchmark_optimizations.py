#!/usr/bin/env python3
"""
Script de benchmark pour comparer les performances avec et sans optimisations
"""

import time
import torch
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json

# Import des modules locaux
import sys
sys.path.append(str(Path(__file__).parent.parent))

from audio_dataset import create_data_loaders
from models.efficientnet_config import create_audio_model


def benchmark_dataloader(loader, name, num_batches=50):
    """Benchmark un DataLoader."""
    print(f"\n📊 Benchmark {name}...")
    
    times = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Warmup
    for i, (data, targets) in enumerate(loader):
        if i >= 5:
            break
        data = data.to(device)
        targets = targets.to(device)
    
    # Benchmark
    start_total = time.time()
    for i, (data, targets) in enumerate(loader):
        if i >= num_batches:
            break
        
        batch_start = time.time()
        data = data.to(device)
        targets = targets.to(device)
        batch_time = time.time() - batch_start
        times.append(batch_time)
    
    total_time = time.time() - start_total
    
    # Statistiques
    times = np.array(times)
    stats = {
        'name': name,
        'total_time': total_time,
        'num_batches': len(times),
        'avg_time_per_batch': np.mean(times),
        'std_time_per_batch': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'batches_per_second': 1.0 / np.mean(times)
    }
    
    print(f"  - Temps total: {total_time:.2f}s")
    print(f"  - Temps moyen/batch: {stats['avg_time_per_batch']*1000:.1f}ms")
    print(f"  - Batches/seconde: {stats['batches_per_second']:.2f}")
    
    return stats


def benchmark_training_step(model, loader, name, num_steps=20, use_amp=False):
    """Benchmark une étape d'entraînement complète."""
    print(f"\n🚀 Benchmark entraînement {name}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    
    times = []
    
    # Warmup
    for i, (data, targets) in enumerate(loader):
        if i >= 5:
            break
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(data)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_total = time.time()
    
    for i, (data, targets) in enumerate(loader):
        if i >= num_steps:
            break
        
        step_start = time.time()
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(data)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        step_time = time.time() - step_start
        times.append(step_time)
    
    total_time = time.time() - start_total
    
    # Statistiques
    times = np.array(times)
    stats = {
        'name': name,
        'use_amp': use_amp,
        'total_time': total_time,
        'num_steps': len(times),
        'avg_time_per_step': np.mean(times),
        'std_time_per_step': np.std(times),
        'steps_per_second': 1.0 / np.mean(times)
    }
    
    print(f"  - Temps total: {total_time:.2f}s")
    print(f"  - Temps moyen/step: {stats['avg_time_per_step']*1000:.1f}ms")
    print(f"  - Steps/seconde: {stats['steps_per_second']:.2f}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark des optimisations")
    parser.add_argument('--data-dir', type=Path, required=True,
                       help="Répertoire contenant les fichiers audio")
    parser.add_argument('--csv-dir', type=Path, default=Path('data/processed/csv'),
                       help="Répertoire contenant les CSV")
    parser.add_argument('--batch-size', type=int, default=32,
                       help="Taille du batch")
    parser.add_argument('--num-batches', type=int, default=50,
                       help="Nombre de batches à tester")
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help="Fichier de sortie pour les résultats")
    
    args = parser.parse_args()
    
    print("🔬 Benchmark des Optimisations NightScan Audio")
    print("=" * 50)
    
    # Informations système
    print("\n💻 Système:")
    print(f"  - PyTorch: {torch.__version__}")
    print(f"  - CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        'benchmarks': {}
    }
    
    # Test 1: DataLoader sans optimisations
    print("\n\n1️⃣ DataLoader SANS optimisations")
    loader_slow = create_data_loaders(
        csv_dir=args.csv_dir,
        audio_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=2,
        persistent_workers=False,
        prefetch_factor=2
    )
    if 'train' in loader_slow:
        results['benchmarks']['dataloader_slow'] = benchmark_dataloader(
            loader_slow['train'], "Sans optimisations", args.num_batches
        )
    
    # Test 2: DataLoader avec optimisations
    print("\n\n2️⃣ DataLoader AVEC optimisations")
    loader_fast = create_data_loaders(
        csv_dir=args.csv_dir,
        audio_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4
    )
    if 'train' in loader_fast:
        results['benchmarks']['dataloader_fast'] = benchmark_dataloader(
            loader_fast['train'], "Avec optimisations", args.num_batches
        )
    
    # Test 3: DataLoader avec cache spectrogrammes
    cache_dir = Path("data/spectrograms_cache")
    if cache_dir.exists():
        print("\n\n3️⃣ DataLoader AVEC cache spectrogrammes")
        loader_cached = create_data_loaders(
            csv_dir=args.csv_dir,
            audio_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=8,
            spectrogram_dir=cache_dir,
            persistent_workers=True,
            prefetch_factor=4
        )
        if 'train' in loader_cached:
            results['benchmarks']['dataloader_cached'] = benchmark_dataloader(
                loader_cached['train'], "Avec cache", args.num_batches
            )
    
    # Test 4: Entraînement sans AMP
    if torch.cuda.is_available() and 'train' in loader_fast:
        print("\n\n4️⃣ Entraînement SANS Mixed Precision")
        model = create_audio_model(num_classes=6, model_name='efficientnet-b0')
        results['benchmarks']['training_no_amp'] = benchmark_training_step(
            model, loader_fast['train'], "Sans AMP", num_steps=20, use_amp=False
        )
        
        # Test 5: Entraînement avec AMP
        print("\n\n5️⃣ Entraînement AVEC Mixed Precision")
        model = create_audio_model(num_classes=6, model_name='efficientnet-b0')
        results['benchmarks']['training_with_amp'] = benchmark_training_step(
            model, loader_fast['train'], "Avec AMP", num_steps=20, use_amp=True
        )
    
    # Résumé des résultats
    print("\n\n📈 RÉSUMÉ DES PERFORMANCES")
    print("=" * 50)
    
    if 'dataloader_slow' in results['benchmarks'] and 'dataloader_fast' in results['benchmarks']:
        slow = results['benchmarks']['dataloader_slow']['batches_per_second']
        fast = results['benchmarks']['dataloader_fast']['batches_per_second']
        print(f"\nDataLoader:")
        print(f"  - Sans optimisations: {slow:.2f} batches/s")
        print(f"  - Avec optimisations: {fast:.2f} batches/s")
        print(f"  - Amélioration: {fast/slow:.2f}x")
    
    if 'dataloader_cached' in results['benchmarks'] and 'dataloader_slow' in results['benchmarks']:
        cached = results['benchmarks']['dataloader_cached']['batches_per_second']
        slow = results['benchmarks']['dataloader_slow']['batches_per_second']
        print(f"\nAvec cache spectrogrammes:")
        print(f"  - Performance: {cached:.2f} batches/s")
        print(f"  - Amélioration vs baseline: {cached/slow:.2f}x")
    
    if 'training_no_amp' in results['benchmarks'] and 'training_with_amp' in results['benchmarks']:
        no_amp = results['benchmarks']['training_no_amp']['steps_per_second']
        with_amp = results['benchmarks']['training_with_amp']['steps_per_second']
        print(f"\nMixed Precision (AMP):")
        print(f"  - Sans AMP: {no_amp:.2f} steps/s")
        print(f"  - Avec AMP: {with_amp:.2f} steps/s")
        print(f"  - Amélioration: {with_amp/no_amp:.2f}x")
    
    # Sauvegarder les résultats
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\n✅ Résultats sauvegardés dans: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())