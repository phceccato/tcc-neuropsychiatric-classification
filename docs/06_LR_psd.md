# LR_psd — regressão logística sobre a PSD em bandas: resultados

As AUCs por pessoa da regressão logística sobre as 76 features de banda são **0,72 na SZ** (IC de 95% de 0,52 a 0,90) e **0,87 no MDD** (0,76 a 0,96). No MDD o IC fica bem acima de 0,5. Na SZ o limite inferior mal passa de 0,5, e o resultado é só um indício fraco. Não há teste contra o acaso: a incerteza é o IC.

- Código: [pipeline/e06_LR_psd/lr.py](../pipeline/e06_LR_psd/lr.py), com as métricas de [avaliacao.md](avaliacao.md)
- Checagem por sessão: [pipeline/e06_LR_psd/checagem_lote.py](../pipeline/e06_LR_psd/checagem_lote.py), post-hoc, descritiva, sem retreinar a LR da tabela principal (testes em `tests/e06_LR_psd/test_checagem_lote.py`)
- Duração: ~10 s (CPU)
- Saídas em `outputs/06_LR_psd/`:
  - `predicoes.csv`: probabilidade fora do fold de cada pessoa, em cada repetição;
  - `resumo.md`;
  - `checagem_lote.md`: a checagem da potência relativa por sessão (abaixo).

```bash
venv/bin/python -m pipeline.e06_LR_psd.lr
venv/bin/python -m pipeline.e06_LR_psd.checagem_lote   # ~10 s
```

## Modelo

- **Classificador:** regressão logística L2 com C = 1 e peso de classe (`class_weight="balanced"`).
- **Entrada:** média das features nas janelas aceitas de cada pessoa. As pessoas de treino usam a grade com 50% de overlap, e as de teste a grade sem overlap.
- **Normalização:** z-score por feature, com estatísticas só do treino de cada fold.
- **Validação:** as partições de [05_folds.md](05_folds.md), 5 folds × 10 repetições.
- **Sem teste contra o acaso:** o trabalho não usa p-valores de permutação (nem para o `DTSF_FA`). A incerteza é o IC de 95% da AUC por pessoa. Um teste de permutação da LR existiu até a revisão de 16/11/2026 (SZ p = 0,056; MDD p = 0,001) e foi removido por coerência; as AUCs e as previsões não mudaram.

## Resultados

| Conjunto | Features | AUC por pessoa [IC 95%] | AUC por repetição | Acurácia balanceada | Sensibilidade | Especificidade | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SZ | absolutas | 0,724 [0,520; 0,903] | 0,717 ± 0,046 | 0,714 | 0,786 | 0,643 | 0,713 |
| SZ | relativas | 0,668 [0,459; 0,857] | 0,664 ± 0,056 | 0,571 | 0,643 | 0,500 | 0,569 |
| MDD | absolutas | 0,868 [0,761; 0,959] | 0,859 ± 0,021 | 0,803 | 0,828 | 0,778 | 0,803 |
| MDD | relativas | 0,879 [0,774; 0,962] | 0,875 ± 0,011 | 0,838 | 0,862 | 0,815 | 0,839 |

Como ler a tabela:
- **AUC por pessoa:** calculada sobre a média da probabilidade de cada pessoa nas 10 repetições. O IC vem de um bootstrap estratificado com 2000 reamostragens.
- **AUC por repetição:** média ± desvio das 10 AUCs.
- **Média ± desvio:** o `resumo.md` traz também uma tabela com a média ± desvio, nas 10 repetições, da AUC e das secundárias, calculadas em cada repetição com os 5 folds juntos ([avaliacao.md](avaliacao.md#média--desvio-nas-repetições)).
- **Secundárias:** calculadas no limiar 0,5 e só descritivas.

## Leitura

- **SZ:** a AUC de 0,72 tem IC de 0,52 a 0,90, largo como o protocolo previa para 28 pessoas, com o limite inferior quase em 0,5. O resultado é sensível à partição: com a semente antiga (ver abaixo), a mesma LR dava 0,78. Com 28 pessoas, a diferença entre as duas sementes está dentro da variação esperada (o desvio entre as repetições é de 0,05), e a leitura para a SZ é "indício fraco", sem afirmar que supera o acaso.
- **MDD, potência relativa:** ela vai tão bem quanto a absoluta (0,879 contra 0,868). Isso mostra que o resultado não depende só do ganho do aparelho, mas **não mostra que ele é livre de efeito de lote**: a normalização não remove diferenças de ritmo entre sessões (próxima seção). A AUC do MDD continua sujeita à sessão de gravação (AUC da data 0,93, [09_investigacao.md](09_investigacao.md)).
- **Uso:** é a referência mais simples do projeto. O DTSF está em [07_DTSF.md](07_DTSF.md), e os baselines clássicos sobre as entradas do DTSF em [08_classicos.md](08_classicos.md).

## Checagem da potência relativa por sessão

A proposta usa a potência relativa (banda ÷ soma das 4 bandas) como checagem do MDD: se a absoluta vencesse muito só no MDD, a suspeita seria o ganho do amplificador. Essa checagem **só pode detectar um efeito de escala**. A divisão pela potência total cancela um ganho multiplicativo, mas não muda a forma do espectro. Um ritmo dominante em ~5 Hz continua deslocando a fração de potência para teta. Além disso, comparar duas LR (absoluta contra relativa) não separa lote de diagnóstico: as duas leem a mesma sessão, e a sessão de jan/2013 tem só controles.

`checagem_lote.py` estratifica as 56 pessoas do MDD pelas sessões reconstruídas da data do EDF: S1 (jul–ago/2011, 9 HC), S2 (12–14/jan/2013, 14 HC) e S3 (abr–jul/2013, 29 pacientes e 4 HC). Os números são post-hoc e descritivos.

**O 0,43 vs 0,20 vem de uma sessão.** Potência relativa em teta (média ± desvio entre pessoas):

| Sessão | Classe | n | Teta absoluta (log10 µV²) | Teta relativa |
| --- | --- | --- | --- | --- |
| S1 | HC | 9 | 0,66 ± 0,16 | 0,17 ± 0,05 |
| **S2 (jan/2013)** | HC | 14 | **1,56 ± 0,35** | **0,64 ± 0,17** |
| S3 | HC | 4 | 0,51 ± 0,05 | 0,25 ± 0,09 |
| S3 | MDD | 29 | 0,52 ± 0,32 | 0,20 ± 0,11 |

A média de 0,43 dos controles é a mistura de 0,64 (S2, n = 14) com 0,17 a 0,25 (S1 e S3, n = 13). Fora da S2, a teta relativa dos controles é igual à dos pacientes (AUC da teta relativa sem a S2: 0,49). O efeito persiste depois da normalização (de 1,56 contra ~0,55 em log10 passa a 0,64 contra ~0,20) e está ligado ao lote: os 14 controles da S2 têm também delta relativa de 0,04 (contra 0,10 a 0,17 nos outros), alfa de 0,23 (contra 0,37 a 0,56) e beta de 0,09. Esse perfil é o de um ritmo dominante em ~5 Hz, que a divisão pela soma das bandas não anula, porque o ritmo ocupa a maior parte da potência total.

**A LR de 76 bandas não depende só dessa sessão**, mas a sessão continua confundida com a classe. Retreinando a LR nas mesmas partições sem as 14 pessoas da S2:

| Features | Pessoas | AUC por pessoa [IC 95%] |
| --- | --- | --- |
| absolutas | todas (tabela principal) | 0,868 [0,761; 0,959] |
| absolutas | sem a S2 | 0,859 [0,703; 0,984] |
| relativas | todas (tabela principal) | 0,879 [0,774; 0,962] |
| relativas | sem a S2 | 0,894 [0,743; 1,000] |

A AUC quase não muda, então o resultado do MDD não vem só do ritmo de 5 Hz. Mas sem a S2 os 13 controles restantes ainda estão quase todos fora da sessão dos pacientes (9 na S1, de 2011, e só 4 na S3), e a beta relativa, que separa bem (AUC 0,85 sem a S2; 0,81 dentro da S3, com 4 controles), pode refletir também sessão ou equipamento. O IC fica mais largo (0,70 a 1,00). Por isso a conclusão não é que o MDD está livre de efeito de lote, e sim que o efeito mais óbvio (S2) não explica sozinho a AUC.

**Conclusões que o texto deve usar.**
- A normalização por potência total elimina o ganho do aparelho, mas **não elimina diferenças de ritmo entre lotes**.
- A potência relativa **não é uma checagem de lote**. Ela não deve ser citada como evidência de que o MDD está livre de efeito de aquisição.
- A diferença de teta entre controles e pacientes do MDD (0,43 contra 0,20) é um **artefato de composição**: ela vem de uma sessão de controles. Não deve ser lida como achado clínico.
- A AUC do MDD (e a da data, 0,93) continua a ser tratada como possivelmente inflada por sessão, como limitação central do trabalho. Nenhum dado ou protocolo foi alterado por esta checagem.

## Semente das partições

Uma versão anterior deste documento trazia SZ 0,781 (com p = 0,010 de um teste de permutação, depois removido). Esses números vinham de uma rodada com a semente antiga (`SEMENTE = 20261011`), anterior à troca de todas as sementes para 20261108 (partições, bootstrap e, na época, as permutações). Os resultados acima são os da semente atual, a do protocolo. Reexecutando o mesmo código com a semente antiga, só em pasta temporária, reproduzem-se exatamente os números antigos (SZ absolutas 0,781 [0,592; 0,923]; SZ relativas 0,704; MDD 0,865 e 0,884). A troca de semente é a única causa da diferença; o protocolo (modelo, features, normalização, 5 × 10 folds) não mudou, e os números antigos não substituem os atuais. Com a semente atual a execução é determinística: duas rodadas dão `predicoes.csv` idênticos byte a byte.
