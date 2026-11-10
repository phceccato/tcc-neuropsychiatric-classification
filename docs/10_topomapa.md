# Mapas topográficos por pessoa

Figura com quatro mapas topográficos (delta, teta, alfa e beta) da potência espectral de **uma única pessoa**, no estilo de mapa de qEEG: escala de cor `jet`, isolinhas, eletrodos marcados, contorno da cabeça com nariz e orelhas e uma barra de cor por banda. É uma visualização descritiva: não treina modelo, não testa hipótese e não altera nenhuma etapa anterior nem os dados.

- Código: [pipeline/e10_topomapa/](../pipeline/e10_topomapa/)
  - `topomapa.py`: a figura de uma pessoa;
  - `todos.py`: as 84 pessoas, em pastas por dataset, classe e sessão.
- Testes: `tests/e10_topomapa/`
- Entrada: `outputs/04_features/psd.npz` e `outputs/03_janelas/janelas.csv` (só leitura)
- Saída:
  - `topomapa.py`: `outputs/10_topomapa/topomapa_<pessoa>_<escala>.png`, a 300 DPI;
  - `todos.py`: `outputs/10_topomapa/<dataset>/<classe>/<sessão>/<absoluta|relativa>/topomapa_<pessoa>_<escala>.png` e o índice `outputs/10_topomapa/indice_<escala>.csv`.
- Execução: as 84 pessoas na escala `log` (11 s com 8 processos).

```bash
venv/bin/python -m pipeline.e10_topomapa.topomapa --pessoa MDD-MDD-S27                    # log10 da potência absoluta (padrão)
venv/bin/python -m pipeline.e10_topomapa.topomapa --pessoa MDD-MDD-S27 --escala relativa  # % da potência de 2–30 Hz
venv/bin/python -m pipeline.e10_topomapa.topomapa --pessoa MDD-MDD-S27 --escala absoluta  # µV²
```

```bash
venv/bin/python -m pipeline.e10_topomapa.todos                     # as 84 pessoas (retoma; --refazer gera de novo)
venv/bin/python -m pipeline.e10_topomapa.todos --escala relativa   # outra escala, em arquivos separados
```

Pré-requisitos: as etapas 02, 03 e 04.

## Organização das pastas (`todos.py`)

`<dataset>/<classe>/<sessão>/<absoluta|relativa>/`:
- **dataset:** `Schizophrenia_dataset` (SZ) ou `Depression_dataset` (MDD);
- **classe:** `HC` ou o diagnóstico (`SZ`, `MDD`), da coluna `grupo` do manifesto;
- **sessão:** as sessões reconstruídas na etapa 09 (`pessoas_por_dataset`: sessão nova quando duas datas consecutivas distam mais de 30 dias). O nome traz o número e o período, sem "/" (por exemplo, `sessao_2_jun-jul2003`), e o número é o do dataset inteiro, então a sessão 3 do MDD tem o mesmo nome nas pastas de HC e de MDD;
- **escala:** `absoluta/` guarda os mapas em log10 da potência absoluta (`_log.png`; também os de µV² lineares, `_absoluta.png`, se gerados) e `relativa/` os de potência relativa (`_relativa.png`). A média da sessão fica na subpasta da escala dela.

| Dataset | Classe | Sessão | Pessoas |
| --- | --- | --- | --- |
| Schizophrenia_dataset | HC | sessao_2_jun-jul2003 | 14 |
| Schizophrenia_dataset | SZ | sessao_1_mar2003 | 1 |
| Schizophrenia_dataset | SZ | sessao_3_out2003 | 2 |
| Schizophrenia_dataset | SZ | sessao_4_ago2004 | 2 |
| Schizophrenia_dataset | SZ | sessao_5_abr2005 | 1 |
| Schizophrenia_dataset | SZ | sessao_6_ago2006 | 3 |
| Schizophrenia_dataset | SZ | sessao_7_out2006 | 3 |
| Schizophrenia_dataset | SZ | sessao_8_fev2007 | 1 |
| Schizophrenia_dataset | SZ | sessao_9_jul2008 | 1 |
| Depression_dataset | HC | sessao_1_jul-ago2011 | 9 |
| Depression_dataset | HC | sessao_2_jan2013 | 14 |
| Depression_dataset | HC | sessao_3_abr-jul2013 | 4 |
| Depression_dataset | MDD | sessao_3_abr-jul2013 | 29 |

As contagens batem com docs/09_investigacao.md, seção 1.

**Médias por sessão.** Cada pasta de sessão tem também `media_<sessão>_<escala>.png`: a média, entre as pessoas da pasta, dos valores [19 canais × 4 bandas] de cada pessoa (`medias_por_pasta`), desenhada com a mesma barra comum das figuras individuais. É a média dos valores, não das imagens; na escala `log`, é a média do log10 (média geométrica da potência), e cada pessoa pesa o mesmo, qualquer que seja o número de janelas. Como a média é por pasta, classes não se misturam: a sessão 3 do MDD tem uma média dos 4 HC e outra dos 29 pacientes. São 13 médias; nas sessões com 1 pessoa (SZ, sessões 1, 5, 8 e 9), a média é a própria pessoa. As médias são refeitas a cada execução de `todos.py`.

Leitura (descritiva), teta médio dos 19 canais, em log10 µV²:

| MDD | Pessoas | Teta | Alfa |
| --- | --- | --- | --- |
| HC, sessão 1 (jul–ago/2011) | 9 | 0,69 | 1,26 |
| HC, sessão 2 (jan/2013) | 14 | **1,57** | 1,08 |
| HC, sessão 3 (abr–jul/2013) | 4 | 0,54 | 0,79 |
| MDD, sessão 3 (abr–jul/2013) | 29 | 0,56 | 0,87 |

Os HC de jan/2013 têm teta cerca de 10 vezes maior na potência (1 unidade de log10) que os pacientes, a marca de lote vista nas etapas 04 e 09. Os 4 HC gravados na mesma sessão dos pacientes têm teta e alfa parecidos com os deles (0,54 contra 0,56; 0,79 contra 0,87). O `indice_<escala>.csv` lista, por pessoa, o dataset, a classe, a data de gravação, a sessão, as janelas aceitas e o caminho da figura.

## Metodologia

**Potência por banda (reaproveitada da etapa 04, sem recalcular).** Para cada janela de 4 s, `psd.npz` já guarda o log10 da potência absoluta (µV²) de cada canal × banda: Welch com segmentos de 2 s, janela de Hann e 50% de overlap, resolução de 0,5 Hz, sobre o sinal em µV filtrado em 2–30 Hz (etapa 03). As bandas são as de `pipeline/e04_features/psd.py`:

| Banda | Faixa |
| --- | --- |
| Delta | [2, 4) Hz |
| Teta | [4, 8) Hz |
| Alfa | [8, 13) Hz |
| Beta | [13, 30] Hz |

As bandas semiabertas (o limite superior fica de fora, exceto em beta) evitam contar o mesmo bin em duas bandas.

**Seleção da pessoa.** Só as linhas de `janelas.csv` com o `subject_uid` pedido e `rejeitada` = False. O script confere que `psd.npz` e `janelas.csv` estão alinhados linha a linha e que a ordem das 76 features é a de `psd.py`. Também para com erro se a pessoa não existe, se não tem janela aceita ou se há valor não finito. Nada de outra pessoa entra no mapa, nem para normalizar a escala.

**Agregação das janelas.** Em cada canal e banda, a potência absoluta das janelas (10^valor, em µV²) é somada numa média aritmética. A média é feita na potência linear, não no log, para não virar média geométrica. A escala do mapa sai dessa média e fica escrita no título da figura (`--escala`):
- `log` (padrão): log10 da potência absoluta média, em log10(µV²);
- `absoluta`: potência absoluta média, em µV²;
- `relativa`: potência média da banda dividida pela soma das 4 bandas no mesmo canal, em %.

**Coordenadas 10–20.** Os 19 canais (`CANAIS`, etapa 01) usam a montagem do próprio pipeline, `colin27_1020` (`MONTAGEM`, etapa 02, a mesma da interpolação de canais), que traz os nomes antigos T3–T6. Nenhuma coordenada é inventada. Duas transformações rígidas, que não mudam a distância entre eletrodos (`posicoes_niveladas`), ajustam a projeção:
1. **Rotação em torno do eixo x até Fpz e Oz ficarem na mesma altura.** Na colin27 o plano Fpz–Oz é inclinado, e sem isso os eletrodos frontais caíam fora do contorno e os occipitais dentro.
2. **Translação para o centro da cabeça ficar na origem.** O centro é definido por Fpz, Oz, T7 e T8, o critério do `sphere="eeglab"` do MNE. O MNE mede até onde estender o mapa a partir da origem (`mask_scale` em `mne/viz/topomap.py`), então um centro deslocado fazia o mapa passar do contorno.

O contorno da cabeça (`raio_contorno`) é o círculo que envolve o eletrodo mais externo com 3% de folga. Como a cabeça do colin27 não é esférica, os eletrodos periféricos caem até ~9% fora do círculo que passa por Fpz, Oz, T7 e T8. O raio só muda o desenho, não a posição projetada dos eletrodos.

**Interpolação e desenho.** `mne.viz.plot_topomap` (MNE 1.13.2): interpolação cúbica entre os 19 eletrodos, extrapolada até o contorno (`extrapolate="head"`), marcação dos eletrodos e 6 isolinhas em níveis fixos, igualmente espaçados dentro da barra.

**Escala de cor comum (padrão).** Em cada banda, a barra vai do mínimo ao máximo dos 19 canais de **todas as 84 pessoas** (`limites_comuns`), igual em todas as figuras. Assim a mesma cor e a mesma isolinha significam o mesmo valor em qualquer figura, e dá para comparar pessoas, classes e sessões. Os outros participantes só definem os limites da barra; os valores desenhados são só da pessoa. As bandas continuam com barras diferentes entre si, porque a potência de delta, teta, alfa e beta difere muito. Limites na escala `log` (log10 µV², gravados em `outputs/10_topomapa/limites_log.csv`):

| Banda | Mínimo | Máximo |
| --- | --- | --- |
| Delta | −0,486 | 1,455 |
| Teta | −0,076 | 2,534 |
| Alfa | −0,230 | 2,521 |
| Beta | −0,179 | 1,932 |

São o mínimo e o máximo exatos, sem corte por percentil: nenhuma cor satura, ao custo de menos contraste dentro de cada mapa. Os extremos vêm de poucas pessoas (por exemplo, teta alto em controles da sessão de jan/2013 do MDD). `topomapa.py --escala-individual` volta à barra pelo mínimo–máximo da própria pessoa, com sufixo `_individual` no arquivo.

## Diferença em relação à figura de referência

A referência tem escalas simétricas em torno de zero (por exemplo, −6,0 a 6,0), típicas de mapas de **z-score contra um banco normativo**: cada canal comparado com a média de uma população de referência. Isso exigiria comparar a pessoa com outras pessoas, o que o pedido excluiu ("não misturar participantes"). Aqui o mapa mostra a potência da própria pessoa, e a barra é a faixa observada no conjunto das 84 pessoas. A referência também usa bandas diferentes (por exemplo, delta 1,0–3,5 Hz); aqui ficam as bandas do pipeline, porque o sinal foi filtrado em 2–30 Hz.

## Conferências

- Os máximos dos mapas batem com os valores por canal: em MDD S27, delta em F8, teta em T5, alfa em T5/O2 e beta em O1, nas posições esperadas, com a direita do lado direito e o nariz para cima.
- Projeção: Cz a 2% do raio do centro e todos os eletrodos dentro do contorno (o mais externo a 0,102 m, com contorno de 0,105 m); o corte do mapa coincide com o contorno.
- Testes:
  - só janelas aceitas da pessoa escolhida;
  - média feita na potência linear;
  - potência relativa somando 100%;
  - distâncias entre eletrodos iguais às da montagem;
  - contorno envolvendo os eletrodos;
  - PNG a 300 DPI.

## Potência relativa

`todos.py --escala relativa` gerou as mesmas 84 figuras e 13 médias com a potência relativa (banda ÷ soma das 4 bandas no canal, em %), em arquivos `_relativa.png` ao lado dos `_log.png`. A barra também é comum às 84 pessoas em cada banda (`limites_relativa.csv`): delta 0,4–55,1%, teta 2,5–95,3%, alfa 2,0–90,2%, beta 0,9–78,6%. O máximo de teta (95,3%) é de MDD-HC-S19, controle de jan/2013.

Média dos 19 canais por pasta (descritivo):

| Grupo | Pessoas | Delta | Teta | Alfa | Beta |
| --- | --- | --- | --- | --- | --- |
| SZ, HC, jun–jul/2003 | 14 | 16,7% | 20,4% | 43,8% | 19,1% |
| MDD, HC, jul–ago/2011 | 9 | 10,7% | 16,4% | 58,2% | 14,8% |
| MDD, HC, jan/2013 | 14 | 4,0% | **62,8%** | 23,9% | 9,2% |
| MDD, HC, abr–jul/2013 | 4 | 16,7% | 23,5% | 41,3% | 18,5% |
| MDD, MDD, abr–jul/2013 | 29 | 10,3% | 19,6% | 38,9% | 31,3% |

Sem o efeito de ganho, a marca do lote de jan/2013 continua: o teta é 63% da potência, contra 16–24% nos outros grupos, e o alfa cai para 24%. Isso confirma que o lote difere no formato do espectro, não só no nível.
