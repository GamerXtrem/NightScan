# Première connexion SSH sur un VPS Cloud / VPS Lite Infomaniak

Ce guide résume les étapes nécessaires pour se connecter pour la première fois à un serveur VPS Infomaniak.

## Ouvrir une session root

Dans votre terminal local ou un outil tel que *PuTTY* (Windows) exécutez :

```bash
sudo -i
```

Cette commande ouvre une session interactive avec les droits **root**.

## Connexion depuis macOS ou Linux

Utilisez la commande :

```bash
ssh -i <chemin/vers/cle_privee> <utilisateur>@<adresse_ip>
```

- `<chemin/vers/cle_privee>` : chemin vers votre clé privée (droits `0700` recommandés).
- `<utilisateur>` : nom d'utilisateur par défaut selon la distribution (voir tableau ci-dessous).
- `<adresse_ip>` : adresse IPv4 du serveur indiquée dans le Manager.

Exemple :

```bash
ssh -i c:/path/key ubuntu@192.168.1.1
```

Si un message `WARNING: UNPROTECTED PRIVATE KEY FILE!` apparaît, corrigez les droits :

```bash
chmod 400 <chemin/vers/cle_privee>
```

## Connexion depuis Windows

Windows ne dispose pas nativement d'un client SSH complet. Deux options :

1. Activer le *shell Bash* (Windows 10 minimum).
2. Utiliser les logiciels gratuits **PuTTY** et **PuTTYgen**.

### Conversion de la clé privée

- Ouvrez **PuTTYgen** et cliquez sur **Load** pour charger votre clé privée.
- Sauvegardez-la ensuite via **Save private key**.

### Paramètres PuTTY

- Sous **Session** :
  - **HostName** : adresse IPv4 du serveur.
  - **Port** : laissez `22` (par défaut).
  - **Connection type** : choisissez **SSH**.
- Sous **Connection / SSH / Auth** :
  - Sélectionnez la clé privée générée via PuTTYgen avec **Browse**.
- Cliquez sur **Open** pour ouvrir le terminal puis saisissez votre nom d'utilisateur.

## Utilisateurs par défaut

| Distribution Linux | Utilisateur |
| ------------------ | ---------- |
| AlmaLinux | `almalinux` |
| Arch Linux | `arch` |
| CentOS | `cloud-user` |
| Debian ≤ 7 | `root` |
| Debian ≥ 8 | `debian` |
| Fedora | `fedora` |
| FreeBSD | `freebsd` |
| Ubuntu | `ubuntu` |
| OpenBSD | `openbsd` |
| openSUSE Leap 15 | `opensuse` |
| openSUSE 42 | `root` |
| RancherOS | `rancher` |
| SUSE Linux Enterprise Server | `root` |


