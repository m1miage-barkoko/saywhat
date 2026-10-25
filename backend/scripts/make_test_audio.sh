#!/usr/bin/env bash
# Regenerates tests/data/dialogue_fr.m4a: a ~30 s French dialogue between two
# synthetic voices, with light background noise, encoded like a phone recording.
#
# Requires macOS (`say` with the French voices "Thomas" and "Amélie") and ffmpeg.
# The generated file is committed, so other platforms never need to run this.
# To test with a real recording instead, drop any 2-speaker French file into
# tests/data/ and point the end-to-end test at it (see tests/test_pipeline_e2e.py).
set -euo pipefail

cd "$(dirname "$0")/.."
out="tests/data/dialogue_fr.m4a"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

voice_a="Thomas"   # fr_FR, male
voice_b="Amélie"   # fr_CA, female

lines=(
  "A|Bonjour Claire, alors pour la livraison de mardi, on en est où ?"
  "B|Bonjour. Oui, il faudra prévoir deux palettes, et le camion arrive vers dix heures."
  "A|D'accord. Est-ce que le client a confirmé l'adresse du nouvel entrepôt ?"
  "B|Pas encore. Je l'appelle cet après-midi et je vous envoie un mail."
  "A|Parfait. N'oubliez pas la facture, le montant est de mille deux cents euros."
  "B|C'est noté. On se rappelle jeudi pour faire le point."
  "A|Très bien, merci et bonne journée."
  "B|Merci, à jeudi."
)

concat_list="$tmp/list.txt"
: > "$concat_list"
# 0.5 s of silence between turns, like a natural conversation.
ffmpeg -nostdin -loglevel error -f lavfi -i anullsrc=r=22050:cl=mono -t 0.5 "$tmp/pause.wav"
i=0
for entry in "${lines[@]}"; do
  speaker="${entry%%|*}"
  text="${entry#*|}"
  voice="$voice_a"; [[ "$speaker" == "B" ]] && voice="$voice_b"
  say -v "$voice" -o "$tmp/$i.aiff" "$text"
  ffmpeg -nostdin -loglevel error -i "$tmp/$i.aiff" -ar 22050 -ac 1 "$tmp/$i.wav"
  printf "file '%s'\nfile '%s'\n" "$tmp/$i.wav" "$tmp/pause.wav" >> "$concat_list"
  i=$((i + 1))
done

ffmpeg -nostdin -loglevel error -f concat -safe 0 -i "$concat_list" "$tmp/dialogue.wav"
# Mix in pink noise (street/office hum) ~25 dB below the voices, then encode as
# 44.1 kHz mono AAC — what a phone voice recorder typically produces.
ffmpeg -nostdin -loglevel error -y -i "$tmp/dialogue.wav" \
  -f lavfi -i "anoisesrc=color=pink:amplitude=0.04:seed=42" \
  -filter_complex "[0:a][1:a]amix=inputs=2:duration=first:normalize=0[a]" -map "[a]" \
  -ar 44100 -ac 1 -c:a aac -b:a 64k "$out"

echo "Écrit : $out ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out") s)"
