# Formatage et montage du volume de stockage sur un VPS Cloud

Ce guide explique comment préparer le second disque fourni avec les VPS Infomaniak afin de l'utiliser pour vos données.

## Repérer le volume

Les disques apparaissent sous des noms variables selon la distribution (ex. `/dev/sda`, `/dev/vdb` ...). Pour éviter toute
ambiguïté, il est recommandé d'utiliser l'`UUID` de la partition plutôt que son nom dans le fichier `/etc/fstab`.

```bash
sudo blkid
```

Notez l'`UUID` du volume que vous souhaitez formater et monter.

## Formater le disque

Installez les outils correspondant au système de fichiers voulu. Par exemple pour **XFS** :

```bash
sudo apt install xfsprogs
sudo mkfs.xfs -f /dev/[device]
```

Pour **EXT4** :

```bash
sudo mkfs.ext4 /dev/[device]
```

Remplacez `[device]` par l'identifiant du disque détecté (ex. `/dev/vdb`).

## Monter le volume

1. Créez un répertoire temporaire :
   ```bash
   mkdir /mnt/home
   ```
2. Montez le disque dessus :
   ```bash
   mount /dev/[device] /mnt/home
   ```
3. Copiez le contenu actuel de `/home` :
   ```bash
   rsync -rlptgoDHAX /home/ /mnt/home/
   ```
4. Démontez puis montez définitivement :
   ```bash
   umount /mnt/home
   mount /dev/[device] /home
   rmdir /mnt/home
   ```

En procédant ainsi vous conservez les clés SSH et les droits des utilisateurs.

## Montage automatique au démarrage

Pour que le disque soit monté après un redémarrage, ajoutez son `UUID` à `/etc/fstab` :

```bash
UUID=<votre-UUID> /home xfs noatime,nodiratime,nofail,logbufs=8 0 0
```

Adaptez l'`UUID`, le point de montage et les options selon vos besoins.

