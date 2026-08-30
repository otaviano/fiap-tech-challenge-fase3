#!/usr/bin/env bash
# Baixa o MedQuAD (Ben Abacha & Demner-Fushman, NIH) para data/external/.
#
# O corpus não é versionado neste repositório: são ~73 MB de XML de terceiros,
# com licença própria. O pipeline funciona sem ele — `med-assist data build`
# monta o dataset apenas com o corpus interno se o MedQuAD não estiver presente.
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINO="$RAIZ/data/external/MedQuAD"

if [ -d "$DESTINO" ]; then
  echo "MedQuAD já está em $DESTINO"
  exit 0
fi

mkdir -p "$RAIZ/data/external"
git clone --depth 1 https://github.com/abachaa/MedQuAD.git "$DESTINO"

echo "MedQuAD baixado em $DESTINO ($(find "$DESTINO" -name '*.xml' | wc -l) arquivos XML)."
echo "Rode 'med-assist data build' para regenerar o dataset incluindo o corpus externo."
