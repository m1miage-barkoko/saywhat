# Ce que Verbatim installe, et comment tout retirer

Verbatim garde le maximum de choses **dans le dossier du projet**. Quelques outils ne peuvent pas y
vivre : ils sont listés ici avec la commande pour les retirer.

## 1. Dans le dossier du projet (supprimé avec lui)

| Emplacement | Contenu | Taille (mesurée sur macOS) |
|---|---|---|
| `backend/.venv/` | environnement Python : PyTorch, WhisperX, pyannote… | ≈ 1,1 Go |
| `data/models/huggingface/` | modèles Whisper et pyannote | 3 à 4,5 Go selon les modèles téléchargés |
| `data/models/ctranslate2/` | modèles convertis (`eustlb/distil-large-v3-fr`) | ≈ 1,4 Go par modèle |
| `data/models/torch/` | modèle d'alignement des mots (wav2vec2) | ≈ 370 Mo |
| `data/models/nltk/` | découpage en phrases | ≈ 15 Mo |
| `data/bin/` | programme de débruitage DeepFilterNet | ≈ 30 Mo |
| `data/cache/` | cache de polices de Matplotlib | < 1 Mo |
| `data/cli/` | résultats de la ligne de commande (audio converti, JSON) | selon l'usage |
| `.env` | vos réglages et le jeton Hugging Face | — |

Supprimer un seul modèle : supprimer son dossier `data/models/huggingface/hub/models--<auteur>--<nom>`
(ou `data/models/ctranslate2/<auteur>--<nom>`). Il sera retéléchargé par
`uv run python -m app.download_models` s'il redevient nécessaire.

Verbatim redirige les caches de Hugging Face, PyTorch, NLTK et Matplotlib vers `data/`
(`Settings.use_local_model_dirs`, dans `backend/app/config.py`) : il n'écrit **rien** dans
`~/.cache/huggingface`, `~/.cache/torch`, `~/nltk_data` ou `~/.matplotlib`.

## 2. Hors du projet

### Outils installés avec Homebrew (macOS)

| Formule | Rôle |
|---|---|
| `ffmpeg` | conversion des fichiers audio |
| `uv` | installation de Python et des dépendances |

`brew install ffmpeg` installe aussi les bibliothèques dont ffmpeg a besoin, si elles ne sont pas
déjà présentes. Sur le Mac de développement (25/10/2026), cela a ajouté 11 formules :
`mpg123`, `lame`, `ca-certificates`, `libvpx`, `openssl@4`, `opus`, `sdl2-compat`, `sdl3`, `svt-av1`,
`x264`, `x265` (152 Mo au total avec ffmpeg et uv).

### Fichiers créés par uv

uv ne permet de placer ces deux dossiers dans le projet qu'avec des variables d'environnement à
définir avant chaque commande ; ils restent donc à leur emplacement standard.

| Emplacement | Contenu | Taille | Nécessaire ? |
|---|---|---|---|
| `~/.local/share/uv/python/` | l'interpréteur Python 3.12 utilisé par `backend/.venv` | ≈ 70 Mo | oui, tant que le projet est utilisé |
| `~/.cache/uv/` | cache des paquets téléchargés | ≈ 1 Go | non : `uv cache clean` le vide sans rien casser |

### Compte Hugging Face

Le jeton (`HF_TOKEN`) ne sert qu'au premier téléchargement de pyannote. Une fois les modèles
téléchargés, vous pouvez le retirer de `.env` et le révoquer sur
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## 3. Tout retirer (macOS)

Dans cet ordre (uv doit encore être installé pour l'étape 1) :

```bash
# 1. Interpréteur Python et cache de uv
uv python uninstall 3.12
uv cache clean

# 2. Le projet lui-même (environnement, modèles, données, .env)
rm -rf chemin/vers/verbatim      # par exemple ~/IdeaProjects/verbatim

# 3. ffmpeg et uv, puis les dépendances devenues inutiles
brew uses --installed ffmpeg   # vérifier qu'aucun autre logiciel n'utilise ffmpeg
brew uninstall ffmpeg uv
brew autoremove
```

`brew autoremove` ne retire que les formules installées comme dépendances et dont plus aucun
logiciel n'a besoin : celles qui servent à d'autres programmes restent en place.

Sous Linux ou Windows, remplacez l'étape 3 par la désinstallation de ffmpeg et de uv avec le
gestionnaire utilisé pour les installer (`apt`, `winget`…). Les emplacements uv deviennent
`~/.local/share/uv` et `~/.cache/uv` (Linux) ou `%APPDATA%\uv` et `%LOCALAPPDATA%\uv\cache`
(Windows).
