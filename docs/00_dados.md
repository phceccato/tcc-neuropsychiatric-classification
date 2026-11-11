# Download e conferência dos dados

Baixa as duas bases de EEG das fontes oficiais para `dados/` e confere cada arquivo pelo tamanho e pelo MD5 publicados pela fonte. Serve para montar o mesmo `dados/` em qualquer máquina e para provar que os dados usados são os da fonte, sem alteração.

- Código: [pipeline/e00_dados/baixar.py](../pipeline/e00_dados/baixar.py)
- Lista fixa do que baixar: [pipeline/e00_dados/arquivos.csv](../pipeline/e00_dados/arquivos.csv), com dataset, nome local, bytes, MD5 e URL de cada arquivo (209 linhas).
- Testes: `tests/e00_dados/` (11 testes).
- Saída: `dados/Schizophrenia_dataset/` e `dados/Depression_dataset/` (fora do git).

```bash
venv/bin/python -m pipeline.e00_dados.baixar             # baixa o que falta, confere tudo e imprime o resumo
venv/bin/python -m pipeline.e00_dados.baixar --conferir  # só confere a pasta, sem baixar
venv/bin/python -m pytest tests/e00_dados                # inclui o teste que confere dados/ desta máquina
```

## Fontes

| Base | Fonte e versão | Licença | Arquivos | Bytes |
| --- | --- | --- | --- | --- |
| SZ | Olejarczyk e Jernajczyk (2017), *EEG in schizophrenia*, RepOD versão 1.0, [doi:10.18150/repod.0107441](https://doi.org/10.18150/repod.0107441) | ver a página do RepOD | 28 | 274.341.860 (261,6 MiB) |
| MDD | Mumtaz (2016), *MDD Patients and Healthy Controls EEG Data (New)*, figshare versão 2, [doi:10.6084/m9.figshare.4244171.v2](https://doi.org/10.6084/m9.figshare.4244171.v2) | CC BY 4.0 | 181 | 851.613.696 (812,2 MiB) |

A lista foi gerada das APIs das duas fontes (`--gerar-lista`) e fica fixa no repositório, então o download não depende de a listagem da fonte mudar. Regras:

- **figshare:** a versão 2 lista 193 arquivos, mas 12 aparecem sem MD5 e com o link de download quebrado (6 EC, 3 EO e 3 TASK; ver [01_auditoria.md](01_auditoria.md)). Eles ficam fora da lista, e por isso o MDD tem 181 arquivos.
- **Nome repetido:** "H S15 EO.edf" aparece duas vezes no figshare, com conteúdos diferentes. Quando um nome se repete, o arquivo local leva o id do figshare na frente: `6921143_H S15 EO.edf` e `6921959_H S15 EO.edf`.
- **RepOD:** os 28 EDF da versão 1.0, baixados pela API do Dataverse (`/api/access/datafile/<id>`). O download pela página do RepOD em `.zip` acrescenta um `MANIFEST.TXT`, que o pipeline não usa e que o script não baixa.

## Como o download funciona

- Cada arquivo é gravado em `<nome>.part` e só ganha o nome final depois de conferido o tamanho e o MD5. Um arquivo que não confere é apagado, e o download é tentado de novo (3 tentativas); se continuar errado, o script para com erro.
- É retomável: um arquivo que já existe com o MD5 certo é pulado. Um arquivo local corrompido é baixado de novo.
- O cabeçalho `User-Agent` identifica só o projeto, sem dados pessoais.

## Conferência entre máquinas

O fim de toda execução (e o `--conferir`) confere a pasta inteira contra a lista:

1. cada arquivo da lista existe, com o tamanho e o MD5 da fonte;
2. não há `.edf` a mais nas pastas (um arquivo extra mudaria o que a auditoria lê);
3. o número de arquivos e o total de bytes de cada base batem com a lista.

Depois imprime, por base, uma **impressão digital**: o MD5 da lista ordenada `arquivo:md5`. Duas máquinas com a mesma impressão digital têm exatamente os mesmos arquivos, com o mesmo conteúdo:

| Base | Arquivos | Bytes | Impressão digital |
| --- | --- | --- | --- |
| SZ | 28 | 274.341.860 | `91e07382b04646f85630a69e4638b0e4` |
| MDD | 181 | 851.613.696 | `e98187819f54803c9aeade1885997e52` |

O teste `test_dados_desta_maquina_conferem_com_a_lista` faz a mesma conferência dentro do `pytest` e compara com esses valores. Ele é pulado, com aviso, enquanto `dados/` não existir.

## Resultados

- **`dados/` desta máquina:** os 209 arquivos conferem com a lista (presença, tamanho e MD5), e as impressões digitais são as da tabela acima. A conferência leva cerca de 1 s.
- **Download completo das fontes**, numa pasta vazia: os 209 arquivos baixados em 6 min 33 s (1,1 GB), sem erro nem nova tentativa; a conferência passou com as mesmas impressões digitais, e o MD5 de cada `.edf` é idêntico ao do `dados/` desta máquina.
