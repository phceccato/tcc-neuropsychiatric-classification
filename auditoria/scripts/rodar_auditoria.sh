#!/usr/bin/env bash
# Reexecuta as verificações da auditoria (somente leitura sobre o repositório analisado).
#
# Uso: auditoria/scripts/rodar_auditoria.sh <REPO> <SAIDA> [PYTHON] [RESULTADOS_DFA_R0] [RESULTADOS_CLASSICOS_R0]
#   REPO   cópia do repositório com dados/ (EDF conferidos pelo MD5) e outputs/ das etapas 01–05, 07 (dados e
#          dados_dtsf) e 08 (caracteristicas) já gerados com os comandos do README;
#   SAIDA  pasta onde gravar os .txt e .csv da auditoria;
#   PYTHON interpretador com as versões de requirements.txt (padrão: python);
#   RESULTADOS_DFA_R0 / RESULTADOS_CLASSICOS_R0 (opcionais): saídas de
#          python -m pipeline.e07_DTSF.rodar --modelos D_FA --repeticoes 0 --saida <...>/resultados
#          python -m pipeline.e08_classicos.rodar --modelos LR_combinado KNN_combinado --repeticoes 0 --saida <...>
set -euo pipefail
R=$(realpath "$1"); S=$(realpath -m "$2"); PY=${3:-python}
D=$(dirname "$(realpath "$0")")
mkdir -p "$S"
cd "$R"
run() { local nome=$1; shift; echo "== $nome"; "$PY" "$@" 2>&1 | grep -v -i warning | tee "$S/$nome.txt"; }
run v01_particoes            "$D/v1_particoes.py" "$R"
run v03_duplicatas           "$D/v3_duplicatas.py" "$R"
run v04_espectro_bruto       "$D/v4_espectro_bruto.py" "$R" "$S"
run v05_cabecalhos           "$D/v5_cabecalhos.py" "$R"
run v06_assinaturas          "$D/v6_assinaturas.py" "$R" "$S"
run v07_picos_cz             "$D/v7_eixo_tempo.py" "$R"
run v08_metadados            "$D/v8_metadados.py" "$R" "$S"
run v09_sensibilidade_folds  "$D/v9_sensibilidade_folds.py" "$R"
run v10_lotes_lr_psd         "$D/v10_lotes_lr.py" "$R"
run v11_lote_pela_rede       "$D/v11_lote.py" "$R"
run v13_topografia           "$D/v13_topografia_ecg.py" "$R" "$S"
run v14_sessao_na_classe     "$D/v14_sessao_dentro_da_classe.py" "$R"
if [ -n "${4:-}" ]; then run v12_sessoes_D_FA "$D/v12_dfa_sessoes.py" "$R" "$4" "$S" D_FA; fi
if [ -n "${5:-}" ]; then
  run v12_sessoes_LR_combinado  "$D/v12_dfa_sessoes.py" "$R" "$5" "$S" LR_combinado
  run v12_sessoes_KNN_combinado "$D/v12_dfa_sessoes.py" "$R" "$5" "$S" KNN_combinado
fi
