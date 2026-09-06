#!/bin/zsh
# drive.sh <label> [experiments...]   -- runs experiments in the scratch tree and files the
# outputs under gcfix/<label>/. Flags come from the environment already set.
set -u
SP=/private/tmp/claude-501/-Users-admin-Documents-My-Work-DARCL/df0efdda-14b1-469d-ae55-b24a80cffa93/scratchpad/gcfix
LABEL=$1; shift
if [ $# -gt 0 ]; then EXPS=("$@"); else
  EXPS=(e8_false_mint_null wikipedia_demo e5_order_dependence e7_immediate_vs_deferred e9_support_curve e13_creation_bias)
fi
RUN=$SP/run
OUT=$SP/$LABEL
mkdir -p "$OUT"
export ESCROW_ROOT=$RUN
cd "$RUN/code"
PY=python3
MPL=/private/tmp/claude-501/-Users-admin-Documents-My-Work-DARCL/df0efdda-14b1-469d-ae55-b24a80cffa93/scratchpad/mplenv/bin/python
for e in $EXPS; do
  echo "### $LABEL $e  $(date +%T)"
  /usr/bin/time -p $PY experiments/$e.py > "$OUT/$e.log" 2>&1
  echo "   exit=$?"
done
if [ "${SKIP_LLM:-0}" != "1" ]; then
  echo "### $LABEL llm escrow  $(date +%T)"
  $MPL experiments/llm_graph_formation.py escrow --datasets wikipedia,lazada > "$OUT/llm_escrow.log" 2>&1
  echo "   exit=$?"
  echo "### $LABEL llm score  $(date +%T)"
  $MPL experiments/llm_graph_formation.py score > "$OUT/llm_score.log" 2>&1
  echo "   exit=$?"
fi
mkdir -p "$OUT/results/llm_gf_runs"
cp "$RUN"/results/*.json "$OUT/results/" 2>/dev/null
cp "$RUN"/results/llm_gf_runs/escrow_*.json "$OUT/results/llm_gf_runs/" 2>/dev/null
echo "### $LABEL DONE $(date +%T)"
