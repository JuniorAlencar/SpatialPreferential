#!/usr/bin/env bash
set -euo pipefail

# ========= CONFIG =========
REMOTE="gdrive:SpatialData/Data_Tsallis"

# Ajuste para a pasta onde você gerou os parquets novos (recomendado):
# Ex.: ../../data_rewrite
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

# -------- Parsing do nome (nomenclatura nova) --------
# Exemplo:
# N100000_d2_m2_G2.0_A3.0_seed001to061_nodes.parquet
# N100000_d2_m2_G2.0_A3.0_seed001to061_edges.parquet

get_dim() {
  local f="$1"
  if [[ "$f" =~ _d([0-9]+)_ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo ""
  fi
}

get_ns() { # num_samples
  local f="$1"
  if [[ "$f" =~ _seed001to([0-9]+)_ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo ""
  fi
}

get_kind() {
  local f="$1"
  if [[ "$f" =~ _(nodes|edges)\.parquet$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo ""
  fi
}

# basekey = tudo antes de _seed001toXXX_(nodes|edges).parquet
# (isso inclui N..., d..., m..., G..., A...)
get_basekey() {
  local f="$1"
  echo "$f" | sed -E 's/_seed001to[0-9]+_(nodes|edges)\.parquet$//'
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

# Lista remote de uma Dim (cache)
load_remote_index_for_dim() {
  local remote_dim="$1"
  local tmp_list="$2"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    : > "$tmp_list"
    return
  fi

  rclone lsf "$remote_dim" --files-only --include "*.parquet" > "$tmp_list" || true
}

# maior seed001to no Drive para (base, kind) dentro de uma Dim
remote_max_ns_for_key() {
  local tmp_list="$1"
  local base="$2"
  local kind="$3"

  local max=0
  while IFS= read -r rf; do
    local rkind rbase rns
    rkind="$(get_kind "$rf")"
    [[ "$rkind" != "$kind" ]] && continue

    rbase="$(get_basekey "$rf")"
    [[ "$rbase" != "$base" ]] && continue

    rns="$(get_ns "$rf")"
    [[ -z "$rns" ]] && continue

    if (( rns > max )); then max="$rns"; fi
  done < "$tmp_list"

  echo "$max"
}

# remove do Drive seed001to menores que keep_ns para (base,kind)
cleanup_remote_smaller_ns_cached() {
  local remote_dim="$1"
  local tmp_list="$2"
  local base="$3"
  local kind="$4"
  local keep_ns="$5"

  while IFS= read -r rf; do
    local rkind rbase rns
    rkind="$(get_kind "$rf")"
    [[ "$rkind" != "$kind" ]] && continue

    rbase="$(get_basekey "$rf")"
    [[ "$rbase" != "$base" ]] && continue

    rns="$(get_ns "$rf")"
    [[ -z "$rns" ]] && continue

    if (( rns < keep_ns )); then
      log "Remote cleanup: deleting $remote_dim/$rf (keeping seed001to=$keep_ns)"
      rclone_run deletefile "$remote_dim/$rf"
    fi
  done < "$tmp_list"
}

# -------- Plano local: mantém só o maior seed001to por (dim, base, kind) --------
# Saída por linha:
# dim|base|kind|ns|abs_path|filename
build_plan() {
  local plan_file="$1"
  : > "$plan_file"

  # Escolhe o maior ns por chave dim|base|kind sem depender do formato de G/A
  find "$LOCAL_BASE" -type f -name "*.parquet" -print0 2>/dev/null \
  | awk -v RS='\0' '
      function bname(p,    t){ t=p; sub(/^.*\//, "", t); return t }
      function get_dim(fn,   m){ return match(fn, /_d([0-9]+)_/, m) ? m[1] : "" }
      function get_ns(fn,    m){ return match(fn, /_seed001to([0-9]+)_/, m) ? m[1] : "" }
      function get_kind(fn,  m){ return match(fn, /_(nodes|edges)\.parquet$/, m) ? m[1] : "" }
      function get_base(fn,  t){
        t = bname(fn)
        sub(/_seed001to[0-9]+_(nodes|edges)\.parquet$/, "", t)
        return t
      }
      {
        f=$0
        fn=bname(f)
        dim=get_dim(fn); ns=get_ns(fn); kind=get_kind(fn)
        if (dim=="" || ns=="" || kind=="") next
        base=get_base(f)
        key=dim "|" base "|" kind
        if (!(key in best) || (ns+0) > (best[key]+0)) {
          best[key]=ns
          best_abs[key]=f
          best_fn[key]=fn
        }
      }
      END {
        for (k in best) {
          split(k, a, "|")
          dim=a[1]; base=a[2]; kind=a[3]
          printf "%s|%s|%s|%s|%s|%s\n", dim, base, kind, best[k], best_abs[k], best_fn[k]
        }
      }
    ' > "$plan_file"
}

main() {
  if [[ ! -d "$LOCAL_BASE" ]]; then
    log "ERRO: LOCAL_BASE não existe: $LOCAL_BASE"
    log "Você está rodando de: $(pwd)"
    exit 1
  fi

  local plan
  plan="$(mktemp)"
  build_plan "$plan"

  local total
  total="$(wc -l < "$plan" | tr -d ' ')"
  if [[ "$total" == "0" ]]; then
    log "ERRO: nenhum parquet bateu no padrão dentro de $LOCAL_BASE"
    log "Esperado: N*_d*_m*_G*_A*_seed001to###_(nodes|edges).parquet"
    rm -f "$plan"
    exit 1
  fi

  log "Total de uploads planejados (maior seed001to por dim+base+kind): $total"
  log "Destino remoto: $REMOTE/Dim_<dim>/"
  [[ "$DRY_RUN" -eq 1 ]] && log "DRY_RUN=1 (nenhum upload/remoção será feito)"

  # Ordena por dim para reduzir listagens no remote
  sort -t'|' -k1,1n -k4,4n "$plan" -o "$plan"

  local done=0
  local current_dim=""
  local remote_tmp=""
  local remote_dim_path=""

  while IFS='|' read -r dim base kind ns abs_file bn; do
    done=$((done + 1))
    progress_bar "$done" "$total"

    # Troca de dim => recarrega índice remoto dessa Dim
    if [[ "$dim" != "$current_dim" ]]; then
      current_dim="$dim"
      remote_dim_path="$REMOTE/Dim_${dim}"
      log "== Indexando remoto: $remote_dim_path =="
      [[ -n "${remote_tmp:-}" ]] && rm -f "$remote_tmp" || true
      remote_tmp="$(mktemp)"
      rclone_run mkdir "$remote_dim_path"
      load_remote_index_for_dim "$remote_dim_path" "$remote_tmp"
    fi

    log "Item: dim=$dim | kind=$kind | seed001to(local)=$ns"
    log "BaseKey: $base"
    log "Arquivo:  $bn"

    # CHECK: se local < max no drive, só passa
    local remote_max
    remote_max="$(remote_max_ns_for_key "$remote_tmp" "$base" "$kind")"

    if (( ns <= remote_max )); then
      log "SKIP: local seed001to=$ns < drive_max=$remote_max para (dim=$dim, kind=$kind, base=$base)"
      echo ""
      continue
    fi

    # Upload (sobrescreve se existir)
    log "UPLOAD: local seed001to=$ns >= drive_max=$remote_max  -> $remote_dim_path/$bn"
    rclone_run copyto "$abs_file" "$remote_dim_path/$bn" \
      --transfers "$TRANSFERS" \
      --checkers "$CHECKERS" \
      --drive-chunk-size "$CHUNK" \
      --fast-list \
      --retries 10 \
      --low-level-retries 20 \
      --stats-one-line \
      -P

    # Cleanup no drive: remove menores que ns
    cleanup_remote_smaller_ns_cached "$remote_dim_path" "$remote_tmp" "$base" "$kind" "$ns"

    echo ""
  done < "$plan"

  [[ -n "${remote_tmp:-}" ]] && rm -f "$remote_tmp" || true
  rm -f "$plan"
  log "Concluído."
}

main "$@"