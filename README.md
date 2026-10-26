# Verbatim

Transcription **locale** de conversations en français, avec séparation des intervenants.

Vous donnez un enregistrement de téléphone (m4a, mp3, amr…), Verbatim rend un texte lisible où
chaque voix est identifiée (`Intervenant 1`, `Intervenant 2`…) :

```
[00:00:00] Intervenant 1 : Bonjour Claire, alors pour la livraison de mardi, on en est où ?
[00:00:04] Intervenant 2 : Bonjour, oui, il faudra prévoir 2 palettes, et le camion arrive vers 10 heures.
```

> **État : jalon 1 / 6** — le pipeline fonctionne en ligne de commande. L'API, l'interface web,
> l'édition et les exports arrivent aux jalons suivants.

## Confidentialité

Les enregistrements contiennent la voix d'autres personnes et des informations confidentielles.

- **Tout s'exécute sur votre ordinateur.** Aucun son, aucun texte n'est envoyé à un service externe.
- Le réseau n'est utilisé que par **une seule commande**, `python -m app.download_models`, qui
  télécharge les modèles une fois. Ensuite, Verbatim force le mode hors ligne de Hugging Face
  (`app/privacy.py`) : il fonctionne sans connexion.
- La **télémétrie est désactivée de force** : pyannote.audio 4 envoie par défaut des statistiques
  d'usage (durée des fichiers, nombre d'intervenants) à `otel.pyannote.ai`, et Hugging Face envoie
  les siennes. Verbatim coupe les deux avant leur chargement (des tests le vérifient).
- Les données sont stockées dans `data/` (configurable). **Ne placez pas ce dossier dans un dossier
  synchronisé** (iCloud Drive, Dropbox, OneDrive…) : vos enregistrements y seraient envoyés.
  Verbatim affiche un avertissement si c'est le cas.

## Installation

Testé sur macOS (Apple M4). Linux et Windows sont prévus (dépendances configurées), mais pas encore
testés — ce sera fait au jalon 6.

1. **Prérequis** : [ffmpeg](https://ffmpeg.org) et [uv](https://docs.astral.sh/uv/) (qui installe
   lui-même la bonne version de Python).

   | Système | Commande |
   |---|---|
   | macOS | `brew install ffmpeg uv` |
   | Linux (Debian/Ubuntu) | `sudo apt install ffmpeg` puis `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
   | Windows | `winget install Gyan.FFmpeg astral-sh.uv` |

2. **Dépendances Python** (≈ 2 Go, dont PyTorch) :

   ```bash
   cd backend
   uv sync
   ```

3. **Jeton Hugging Face** (voir ci-dessous), puis **téléchargement des modèles** (≈ 4 Go, rangés
   dans `data/models/` : rien n'est écrit dans votre dossier personnel) :

   ```bash
   uv run python -m app.download_models
   ```

   Relançable sans risque : ce qui est déjà téléchargé est ignoré.

### Jeton Hugging Face

Le modèle de diarisation [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
est gratuit mais « protégé » : il faut accepter ses conditions d'utilisation.

1. Créez un compte sur [huggingface.co](https://huggingface.co/join).
2. Ouvrez la [page du modèle](https://huggingface.co/pyannote/speaker-diarization-community-1),
   connecté, et acceptez les conditions.
3. Créez un jeton de type **Read** sur [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
4. À la racine du projet : `cp .env.example .env`, puis renseignez `HF_TOKEN=hf_...`.

Le jeton ne sert qu'au téléchargement. Ensuite, le modèle est chargé depuis le disque, sans jeton ni
réseau.

## Utilisation (jalon 1 : ligne de commande)

```bash
cd backend
uv run python -m app.pipeline ~/Downloads/reunion.m4a
```

Le texte s'affiche dans le terminal ; le résultat complet (mots, horodatages, intervenants) est écrit
en JSON dans `data/cli/<nom du fichier>/transcript.json`, avec l'audio converti (`audio.wav`) et
débruité (`audio_denoised.wav`) pour comparer. L'original n'est jamais modifié.

| Option | Effet |
|---|---|
| `--speakers 3` | nombre exact d'intervenants, s'il est connu (améliore beaucoup la séparation) |
| `--min-speakers 2 --max-speakers 4` | ou un intervalle |
| `--no-denoise` | sans débruitage (le débruitage dégrade parfois la transcription) |
| `--model large-v3-turbo` | autre modèle de transcription (voir ci-dessous) |
| `--no-diarize` | sans séparation des intervenants (plus rapide) |
| `-o resultat.json` | emplacement du JSON |

## Choisir un modèle

| Modèle | Précision | Vitesse mesurée (Apple M4, CPU) | Téléchargement |
|---|---|---|---|
| `bofenghuang/whisper-large-v3-french` (défaut) | la meilleure en français | ×2,5 temps réel | 3,1 Go |
| `large-v3-turbo` | très bonne | ×4,5 | 1,6 Go |
| `eustlb/distil-large-v3-fr` | bonne | ×4,6 | 1,5 Go (converti au téléchargement) |
| `large-v3` | très bonne (multilingue) | ≈ ×2,5 | 3,1 Go |

« ×2,5 temps réel » : une heure d'audio est transcrite en ≈ 24 min. Mesuré sur 3 min de dialogue,
modèle déjà chargé, 6 threads, lots de 8. La diarisation et l'alignement s'y ajoutent.

- **GPU NVIDIA** : tout tourne sur le GPU en float16, gardez le modèle par défaut.
- **Mac Apple Silicon** : la diarisation et l'alignement utilisent le GPU (Metal), mais Whisper
  tourne sur le CPU (int8), car son moteur, CTranslate2, n'a pas de backend Metal.
- **CPU seul** : `large-v3-turbo` est deux fois plus rapide pour une précision proche. Sur CPU, le
  gain des modèles « rapides » est limité à ≈ ×2 : ils allègent le décodeur, mais l'encodeur, le
  plus coûteux, reste celui de large-v3.

## Comment ça marche

```
original.m4a ─ffmpeg─► audio.wav (16 kHz mono, volume normalisé)
     └─ffmpeg 48 kHz─► DeepFilterNet3 ─► audio_denoised.wav
                                              │
          ┌───────────────────────────────────┴───────────────────────┐
  pyannote community-1                            faster-whisper (WhisperX, VAD + lots)
  qui parle quand (diarisation exclusive)                     │
          │                                       wav2vec2 : horodatage de chaque mot
          └──────────────► attribution mot → intervenant ◄────┘
                           découpage des phrases aux changements de voix
```

Chaque étape est un module indépendant de `backend/app/pipeline/` derrière une interface commune
(`base.py`) : un moteur (par exemple un service cloud, plus tard et sur option) peut en remplacer un
autre sans toucher au reste.

Quelques choix non évidents (détaillés dans le code) :

- **Diarisation exclusive** : pyannote est appelé directement, car WhisperX n'expose que la
  diarisation classique, où les paroles superposées donnent plusieurs intervenants à la fois. La
  version exclusive attribue un seul intervenant à chaque instant, ce qu'il faut pour donner chaque
  mot à une seule personne (`assign_speakers.py`).
- **Mots non horodatés** : l'alphabet du modèle d'alignement français ne contient pas de chiffres ;
  « 1200 » n'a donc pas d'horodatage et prend l'intervenant du mot voisin.
- **Petites erreurs aux frontières** : un mot isolé attribué à une autre voix au milieu d'une phrase
  est considéré comme une imprécision de la diarisation et rendu à ses voisins.
- **DeepFilterNet en binaire** : le paquet Python `deepfilternet` (dernière version en 2023) exige
  numpy < 2, incompatible avec WhisperX (numpy ≥ 2.1). Verbatim utilise le programme autonome publié
  par les auteurs (même modèle DeepFilterNet3), vérifié par somme de contrôle.

## Versions épinglées

| Paquet | Version | Pourquoi |
|---|---|---|
| whisperx | 3.8.6 | dernière version stable (3.8.7 en release candidate) ; utilise pyannote 4 |
| torch / torchaudio | 2.8.0 | imposé par whisperx 3.8.6 (`~=2.8.0`). Piège `weights_only` de torch ≥ 2.6 : pyannote 4 charge bien ses checkpoints avec torch 2.8 (vérifié sur le modèle VAD ; community-1 à confirmer dès le jeton disponible). |
| pyannote.audio | 4.0.7 | nécessaire pour community-1 et sa diarisation exclusive |
| faster-whisper | 1.2.1 | dernière version ; requise par whisperx ≥ 1.2 |
| ctranslate2 | 4.8.2 | récente mais éprouvée (la 4.8.4 date du jour de l'épinglage) |
| DeepFilterNet | 0.5.6 | dernière version publiée (binaire autonome) |

Toutes les dépendances transitives sont figées dans `backend/uv.lock`.

FFmpeg : n'importe quelle version récente. pyannote avertit que `torchcodec` ne trouve pas FFmpeg 4
à 7 : sans importance, Verbatim lui passe toujours l'audio déjà décodé en mémoire (l'avertissement
est masqué).

## Désinstallation

Tout ce que Verbatim installe, dans le projet et en dehors, et les commandes pour le retirer :
[DESINSTALLATION.md](DESINSTALLATION.md).

## Développement

```bash
cd backend
uv run pytest                 # tests rapides (≈ 4 s)
uv run pytest -m models       # bout en bout avec les vrais modèles (≈ 1 min)
uv run ruff check . && uv run ruff format .
```

`tests/data/dialogue_fr.m4a` est un dialogue de 32 s entre deux voix de synthèse avec un bruit de
fond, généré par `backend/scripts/make_test_audio.sh` (macOS). Pour tester avec un vrai
enregistrement, utilisez simplement la commande ci-dessus.

## Dépannage

| Problème | Solution |
|---|---|
| `ffmpeg est introuvable` | installer ffmpeg (voir Installation), puis rouvrir le terminal |
| `… n'est pas encore téléchargé` | `uv run python -m app.download_models` |
| `Un jeton Hugging Face est nécessaire` / `Accès refusé` | voir « Jeton Hugging Face » ; vérifier que les conditions sont acceptées avec le même compte que le jeton |
| Transcription dégradée sur un fichier bruité | relancer avec `--no-denoise` et comparer |
| Avertissement « dossier synchronisé avec le cloud » | déplacez le projet hors d'iCloud Drive / Dropbox / OneDrive, ou définissez `VERBATIM_DATA_DIR` dans `.env` |
