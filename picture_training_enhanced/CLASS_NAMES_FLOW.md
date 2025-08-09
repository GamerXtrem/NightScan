# 📋 Flux des Noms de Classes dans le Pipeline

## ✅ Confirmation

**OUI, les noms de classes sont correctement propagés depuis les noms des dossiers jusqu'aux résultats finaux.**

## 🔄 Flux Complet

### 1. **Dossiers Sources** 📁
```
data/
├── chat/
├── chien/
├── renard/
├── sanglier/
└── chevreuil/
```
Les **noms des dossiers** définissent directement les noms de classes.

### 2. **Data Preparation** 🔧
```python
# data_preparation.py
self.classes = [d.name for d in subdirs]  # Récupère les noms des dossiers
# Sauvegarde dans dataset_metadata.json
{
    "classes": ["chat", "chien", "renard", "sanglier", "chevreuil"],
    "class_to_idx": {"chat": 0, "chien": 1, ...}
}
```

### 3. **PhotoDataset** 📊
```python
# photo_dataset.py
# Récupération automatique depuis les dossiers train/
class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
self.classes = [d.name for d in class_dirs]

# OU depuis les métadonnées si disponibles
if self.metadata and 'classes' in self.metadata:
    self.classes = self.metadata['classes']
```

### 4. **Training & Checkpoints** 💾
```python
# train_real_images.py
# Les classes sont dans le MetricsTracker
self.metrics_tracker = MetricsTracker(
    num_classes=self.dataset.num_classes,
    class_names=self.dataset.classes  # Utilise les noms récupérés
)

# Sauvegardé dans le checkpoint
checkpoint = {
    'dataset_info': self.dataset.get_data_info()  # Contient 'classes'
}
```

### 5. **Evaluation** 📈
```python
# evaluate_model.py
# ImageFolder récupère automatiquement les classes
dataset = datasets.ImageFolder(root=self.test_dir)
# dataset.classes = ['chat', 'chien', 'renard', ...]

# MetricsTracker utilise ces noms
self.metrics_tracker = MetricsTracker(
    class_names=self.dataset.classes
)
```

### 6. **Résultats Finaux** 📊
Les rapports contiennent les noms explicites :
```
MÉTRIQUES PAR CLASSE
--------------------
Classe               Precision    Recall      F1-Score    Support
chat                 95.2%        93.8%       94.5%       156
chien                92.1%        94.3%       93.2%       142
renard               89.7%        87.4%       88.5%       98
sanglier             91.3%        90.2%       90.7%       134
chevreuil            93.6%        95.1%       94.3%       120
```

### 7. **Export de Modèles** 📦
```python
# export_models.py
# Les classes sont dans les métadonnées d'export
info['classes'] = self.checkpoint['dataset_info'].get('classes', [])

# Pour CoreML (iOS)
classifier_config=ct.ClassifierConfig(
    class_labels=self.model_info.get('classes')  # Noms explicites
)
```

## 🧪 Validation

Un script de test `test_class_names_flow.py` vérifie automatiquement que :
1. ✅ Les noms sont extraits des dossiers
2. ✅ Ils sont propagés dans les métadonnées
3. ✅ Le dataset les récupère correctement
4. ✅ Ils sont sauvegardés dans les checkpoints
5. ✅ L'évaluation les utilise pour les métriques
6. ✅ L'export les conserve pour la production

## 💡 Points Importants

### Cohérence Garantie
- Les noms de classes sont **toujours** tirés de la structure des dossiers
- Ordre alphabétique maintenu pour la reproductibilité
- Mapping classe→index conservé tout au long du pipeline

### Cas d'Usage
```python
# Après entraînement, lors de l'inférence :
prediction = model(image)
class_idx = prediction.argmax()
class_name = checkpoint['dataset_info']['classes'][class_idx]
print(f"Prédiction: {class_name}")  # Affiche "chat", "chien", etc.
```

### Robustesse
- Si métadonnées absentes → récupération depuis ImageFolder
- Si ImageFolder échoue → utilisation des métadonnées
- Double vérification pour garantir la cohérence

## 📝 Exemple Concret

Pour un projet avec ces dossiers :
```
images/
├── cerf/
├── loup/
├── lynx/
└── ours/
```

Les résultats afficheront :
```
Classe: cerf    - Accuracy: 94.2%
Classe: loup    - Accuracy: 91.8%
Classe: lynx    - Accuracy: 89.3%
Classe: ours    - Accuracy: 95.6%
```

**Et non pas** :
```
Classe: 0    - Accuracy: 94.2%
Classe: 1    - Accuracy: 91.8%
```

## ✅ Conclusion

**Les noms de classes sont bien conservés et utilisés partout**, depuis les dossiers sources jusqu'aux rapports finaux, en passant par l'entraînement, l'évaluation et l'export des modèles.