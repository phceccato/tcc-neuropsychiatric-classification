"""Integridade do código compartilhado com o trabalho anterior.

Compara a árvore de trabalho com o último commit (HEAD); é pulado fora de um repositório git.
outputs/ não é versionado, então partições e resultados não entram mais aqui.
"""
import subprocess
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
# pipeline/e07_DTSF saiu da lista quando o código do LSTM (T, E, F) foi removido a pedido do autor
# pipeline/e05_folds e pipeline/e06_LR saíram quando a semente foi trocada a pedido do autor
PROTEGIDOS = ["pipeline/avaliacao"]


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True)


@pytest.fixture(scope="module", autouse=True)
def _repositorio():
    if _git("rev-parse", "HEAD").returncode != 0:
        pytest.skip("fora de um repositório git")


@pytest.mark.parametrize("caminho", PROTEGIDOS)
def test_arquivos_versionados_nao_mudaram(caminho):
    assert _git("diff", "--name-only", "HEAD", "--", caminho).stdout.split() == []
