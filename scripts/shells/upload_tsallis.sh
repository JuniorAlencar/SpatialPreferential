#!/usr/bin/env bash
set -euo pipefail

# ========= CONFIG =========
REMOTE="gdrive:SpatialData/Data_Tsallis"

# Script em: SpatialPreferential/scripts/shells/
# Dados em:  SpatialPreferential/Data_Tsallis/
LOCAL_BASE="../../Data_Tsallis"

TRANSFERS=8
CHECKERS=8
CHUNK="64M"

# 1 = simula (não envia/não apaga); 0 = executa
DRY_RUN=0
# ==========================

log() { printf "[%s] %s\n" "$(date '+%F %T')" "$*"; }

rclone_run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DRY_RUN: rclone $*"
  else
    rclone "$@"
  fi
}

# Extrai dim: ..._dim_4_... -> 4
get_dim() {
  local f="$1"
  if [[ "$f" =~ _dim_([0-9]+)_ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo ""
  fi
}

# Extrai Ns: ..._Ns_30.parquet -> 30
get_ns() {
  local f="$1"
  if [[ "$f" =~ _Ns_([0-9]+)\.parquet$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo ""
  fi
}

# Basekey: remove o sufixo _Ns_XX.parquet
get_basekey() {
  local f="$1"
  echo "$f" | sed -E 's/_Ns_[0-9]+\.parquet$//'
}

progress_bar() {
  local done="$1" total="$2" width=30
  local pct=0
  if (( total > 0 )); then pct=$(( done * 100 / total )); fi
  local filled=$(( pct * width / 100 ))
  local empty=$(( width - filled ))

  printf "["
  printf "%0.s#" $(seq 1 "$filled" 2>/dev/null || true)
  printf "%0.s-" $(seq 1 "$empty" 2>/dev/null || true)
  printf "] %3d%% (%d/%d) faltam %d\n" "$pct" "$done" "$total" "$(( total - done ))"
}

# Remove do remote quaisquer arquivos com mesmo basekey e Ns menor que keep_ns
cleanup_remote_smaller_ns() {
  local remote_dim="$1"     # gdrive:.../Dim_X
  local keep_file="$2"      # filename com maior Ns
  local keep_base="$3"      # basekey
  local keep_ns="$4"        # Ns inteiro

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DRY_RUN: cleanup remote $remote_dim (keep base=$keep_base Ns=$keep_ns)"
    return
  fi

  local tmp
  tmp="$(mktemp)"
  rclone lsf "$remote_dim" --files-only --include "*.parquet" > "$tmp" || true

  while IFS= read -r rf; do
    [[ "$rf" == "$keep_file" ]] && continue

    local rbase rns
    rbase="$(get_basekey "$rf")"
    [[ "$rbase" != "$keep_base" ]] && continue

    rns="$(get_ns "$rf")"
    [[ -z "$rns" ]] && continue

    if (( rns < keep_ns )); then
      log "Remote cleanup: deleting $remote_dim/$rf (keeping Ns=$keep_ns)"
      rclone_run deletefile "$remote_dim/$rf"
    fi
  done < "$tmp"

  rm -f "$tmp"
}

# ===== Passo 1: construir plano global (um upload por basekey+dim, escolhendo maior Ns) =====
# Linha: dim|basekey|Ns|abs_path|filename
build_plan() {
  local plan_file="$1"
  : > "$plan_file"

  # Busca recursiva por parquet
  local found_any=0
  while IFS= read -r f; do
    found_any=1
    local bn dim ns base
    bn="$(basename "$f")"

    dim="$(get_dim "$bn")"
    ns="$(get_ns "$bn")"
    [[ -z "$dim" ]] && continue
    [[ -z "$ns" ]] && continue

    base="$(get_basekey "$bn")"
    printf "%s|%s|%s|%s|%s\n" "$dim" "$base" "$ns" "$f" "$bn" >> "$plan_file"
  done < <(find "$LOCAL_BASE" -type f -name "*.parquet" 2>/dev/null)

  if [[ "$found_any" -eq 0 ]]; then
    log "ERRO: não encontrei nenhum *.parquet em $LOCAL_BASE"
    exit 1
  fi
}

main() {
  if [[ ! -d "$LOCAL_BASE" ]]; then
    log "ERRO: LOCAL_BASE não existe: $LOCAL_BASE"
    log "Você está rodando de: $(pwd)"
    exit 1
  fi

  local raw plan
  raw="$(mktemp)"
  plan="$(mktemp)"

  build_plan "$raw"

  # Selecionar apenas o maior Ns por (dim, basekey)
  # Fazemos isso via awk para não depender de arrays gigantes no bash.
  awk -F'|' '
    {
      dim=$1; base=$2; ns=$3; abs=$4; bn=$5;
      key=dim "|" base;
      if (!(key in best) || ns+0 > best[key]+0) {
        best[key]=ns; best_abs[key]=abs; best_bn[key]=bn;
      }
    }
    END {
      for (k in best) {
        split(k, a, "|");
        dim=a[1]; base=a[2];
        printf "%s|%s|%s|%s|%s\n", dim, base, best[k], best_abs[k], best_bn[k];
      }
    }
  ' "$raw" > "$plan"

  rm -f "$raw"

  local total
  total="$(wc -l < "$plan" | tr -d ' ')"
  if [[ "$total" == "0" ]]; then
    log "ERRO: nenhum arquivo bateu no padrão *_dim_<n>_*_Ns_<num>.parquet"
    rm -f "$plan"
    exit 1
  fi

  log "Total de uploads planejados (maior Ns por dim+prefixo): $total"
  log "Destino remoto: $REMOTE/Dim_<dim>/"
  [[ "$DRY_RUN" -eq 1 ]] && log "DRY_RUN=1 (nenhum upload/remoção será feito)"

  # Ordena por dim e por Ns para ficar organizado
  sort -t'|' -k1,1n -k3,3n "$plan" -o "$plan"

  local done=0

  while IFS='|' read -r dim base ns abs_file bn; do
    done=$((done + 1))

    local remote_dim="$REMOTE/Dim_${dim}"

    progress_bar "$done" "$total"
    log "Transferindo agora: dim=$dim | Ns=$ns"
    log "BaseKey: $base"
    log "Arquivo:  $bn"
    log "Local:    $abs_file"
    log "Remote:   $remote_dim/$bn"

    rclone_run mkdir "$remote_dim"

    # upload do maior Ns (sobrescreve se existir)
    rclone_run copyto "$abs_file" "$remote_dim/$bn" \
      --transfers "$TRANSFERS" \
      --checkers "$CHECKERS" \
      --drive-chunk-size "$CHUNK" \
      --fast-list \
      --retries 10 \
      --low-level-retries 20 \
      --stats-one-line \
      -P

    # remove Ns menores do mesmo prefixo dentro dessa Dim
    cleanup_remote_smaller_ns "$remote_dim" "$bn" "$base" "$ns"

    echo ""
  done < "$plan"

  rm -f "$plan"
  log "Concluído: $done/$total"
}

main "$@"
