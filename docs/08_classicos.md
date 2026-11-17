# Baselines clássicos por modalidade — resultados

Nove baselines clássicos para comparar com o DTSF adaptado: regressão logística, Random Forest e KNN × características temporais, espectrais e combinadas (as duas concatenadas), extraídas das mesmas entradas do DTSF. O plano está em [00_proposta-tcc.md](00_proposta-tcc.md). Não há hipóteses a decidir: a etapa traz o ΔAUC do D_FA contra cada baseline, com IC de 95%; o teste estatístico (DeLong com Holm, contra os 3 combinados) está na etapa 11.

Os resultados ficam em [Resultados](#resultados).

- Código: [pipeline/e08_classicos/](../pipeline/e08_classicos/)
  - `caracteristicas.py`: 475 características temporais e 1.102 espectrais por janela;
  - `treino.py`: os 9 modelos, as grades e a seleção interna por pessoa;
  - `rodar.py`: o runner, com retomada e processos em paralelo; monta o combinado concatenando as 475 temporais e as 1.102 espectrais de cada janela (1.577);
  - `avaliar.py`: as AUCs, o ΔAUC com IC de D_FA − baseline (o modelo principal da etapa 07) para os 9 baselines, e o Δ por repetição.
- Testes: `tests/e08_classicos/`.
- Saídas:
  - `outputs/08_classicos/caracteristicas/`: o cache, com `temporais.npy`, `espectrais.npy` e `extracao.json` (todo o `outputs/` fica fora do git);
  - `outputs/08_classicos/resultados/<dataset>/<modelo>/r<rep>_f<fold>.{csv,json}` e `resumo.md`, que traz a AUC por pessoa com IC e uma tabela com a média ± desvio, nas 10 repetições, da AUC e das secundárias ([avaliacao.md](avaliacao.md#média--desvio-nas-repetições));
  - Nada é gravado em `outputs/07_DTSF/resultados` ou `outputs/06_LR_psd`.

Pré-requisitos: as etapas 03 e 05 e os caches da 07 (`pipeline.e07_DTSF.dados` e `pipeline.e07_DTSF.dados_dtsf`). O `avaliar` também precisa das previsões completas do modelo principal (D_FA) em `outputs/07_DTSF/resultados`; sem elas, ele não dá erro, só marca o D_FA como incompleto e não calcula as comparações.

```bash
venv/bin/python -m pipeline.e08_classicos.caracteristicas                     # cache das características, uma vez (~1 min)
venv/bin/python -m pipeline.e08_classicos.rodar 2>&1 | tee -a outputs/08_classicos/log_classicos.txt            # 900 tarefas (9 modelos × 2 datasets × 5 folds × 10 repetições); retoma
venv/bin/python -m pipeline.e08_classicos.avaliar 2>&1 | tee -a outputs/08_classicos/log_classicos_avaliar.txt   # resumo e ΔAUC do D_FA contra os baselines → outputs/08_classicos/resultados/resumo.md
```

## Extração das características

| Item | Valor |
| --- | --- |
| Linhas de `janelas.csv` | 22.681, das quais 21.872 aceitas. As rejeitadas ficam como NaN e nunca são calculadas |
| (canal, representação) constantes em janelas aceitas | 0 |
| Menor log₁₀ da potência da STFT | −6,16 (o piso de −12 nunca atua) |
| Tempo e memória | 1 min 38 s; 3,0 GB |
| Conferência | A STFT recalculada direto do sinal em µV a 250 Hz bate com o cache (erro máximo 4·10⁻⁶ em 5 janelas sorteadas); assimetria, curtose e desvio batem com o scipy (erro máximo 7·10⁻⁸) |

Descritivo, sem rótulos: o desvio do sinal base tem mediana 0,96 (z-score por pessoa) e a curtose em excesso do sinal base tem mediana 0,08 (P99 4,7).

## Validação mínima do fluxo

12 tarefas (6 modelos × 2 datasets), repetição 0, fold 0, **rótulos embaralhados** (um piloto da época das permutações, que o trabalho não usa mais), com a saída na pasta temporária da sessão, fora de `outputs/`. Ela confere o formato e o fluxo, e não é resultado.
- As previsões cobrem exatamente as pessoas de teste do fold.
- As partes internas usam só as pessoas de treino e validação.
- Cada pessoa de ajuste entrou com 60 janelas: 1.320 janelas de ajuste na SZ e 2.700 no MDD. Era a versão anterior; com 150, são ~3.300 e ~6.400.
- A tarefa mais cara foi o RF_espectral no MDD, com 160 s. O conjunto levou 2 min 46 s com 10 processos, ~530 s de CPU.
- `avaliar` rodado sobre essa pasta marca todos os baselines como incompletos, não calcula as comparações e grava o `resumo.md` só na pasta indicada.

## Revisão final

Feita sem rótulos reais e sem resultados dos baselines.
- **Inferência:** a validação cruzada viola a independência das pessoas que o DeLong supõe; isso está declarado na proposta e em [avaliacao.md](avaliacao.md).
- **Janelas:** foi conferido no código e em teste que o sorteio vale só para o ajuste. A validação interna e o teste usam todas as janelas aceitas sem overlap, como no DTSF. A exposição a janelas em relação ao D foi medida reproduzindo o gerador do D. Com 60 por pessoa, os baselines viam ~20% (SZ) e ~54% (MDD) das janelas distintas do D.
- **150 janelas por pessoa:** os baselines passam a ~51% (SZ) e ~128% (MDD) das janelas distintas do D, com +22–25% de pessoas. Como 46 de 56 pessoas do MDD têm menos de 150 aceitas, entraram pesos por janela, N / (2 · P_c · n_i), no lugar do `class_weight="balanced"`, para cada pessoa e cada classe continuarem pesando igual. Na SZ, os pesos são idênticos ao `class_weight` anterior.

## Verificação dos pesos e da comparabilidade

Feita antes de autorizar a rodada, sem treinar nenhum modelo com dados reais.
- **Os pesos chegam aos classificadores, inclusive na seleção interna:**
  - no scikit-learn 1.8, com o roteamento de metadados desligado (o padrão), `clf__sample_weight` chega ao último passo do Pipeline (LR), e o RF recebe `sample_weight` direto;
  - para LR e RF, o ajuste com pesos é idêntico ao do estimador ajustado à mão com os mesmos pesos e diferente do ajuste sem pesos;
  - em teste, cada um dos 10 ajustes de um fold (3 candidatos × 3 partes internas + o final) recebe os pesos recalculados **só com as pessoas daquele ajuste**: dentro de cada classe as pessoas pesam igual, as duas classes pesam igual e a média é 1.
- **Comparabilidade e vazamento:**
  - em teste, as janelas ajustadas são só das pessoas do ajuste, aceitas e sem repetição, e nenhuma é de quem está na parte de validação interna;
  - as janelas de previsão são idênticas com 60 ou com 150 sorteadas, ou seja, o sorteio não toca a validação nem o teste;
  - a previsão de uma pessoa de teste não muda quando os dados das outras pessoas de teste ou os rótulos do teste mudam.
- **Checagem estrutural nos folds reais**, sem treino, reproduzindo o sorteio, as partes internas e os pesos: 100 folds externos e 400 ajustes, **nenhum problema**.
  - O sorteio tem min(150, aceitas) janelas por pessoa, sem repetição e só aceitas.
  - Nenhuma janela de teste entra no ajuste, e nenhuma pessoa é compartilhada entre ajuste, validação interna e teste.
  - As janelas de previsão do teste são exatamente as aceitas sem overlap.
  - Os pesos respeitam as três propriedades acima. A razão entre o maior e o menor peso chega a 1,81 no MDD (a pessoa com 83 janelas) e fica em ≤ 1,14 na SZ, onde é só o desequilíbrio de classes (11 × 12 pessoas), igual ao `class_weight` antigo.

## Seleção na borda das grades (limitação declarada)

As grades são pequenas e foram fixadas antes de qualquer resultado, para limitar o sobreajuste de seleção ([00_proposta-tcc.md](00_proposta-tcc.md)): LR com C ∈ {0,01; 0,1; 1}, RF com `min_samples_leaf` ∈ {1; 5; 20} e KNN com k ∈ {11; 31; 101}. Em alguns modelos, a seleção interna escolhe o valor da **borda** da grade em boa parte dos 50 folds. Contagem nos `.json` (rodada de 9/11, semente 20261108):

| Dataset | Modelo | Valor da borda | Folds (de 50) |
| --- | --- | --- | --- |
| MDD | `LR_temporal` | C = 0,01 (o menor) | 42 |
| SZ | `KNN_temporal` | k = 11 (o menor) | 39 |
| SZ | `KNN_espectral` | k = 11 (o menor) | 38 |
| MDD | `KNN_temporal` | k = 101 (o maior) | 32 |
| MDD | `RF_temporal` | `min_samples_leaf` = 20 (o maior) | 31 |

Nos demais modelos a escolha se espalha pelos três valores. Quando o melhor valor está na borda, o ótimo pode estar fora da grade, e o modelo pode estar um pouco subajustado. Isso poderia **subestimar os clássicos** e favorecer a leitura de que o `D_FA` é melhor.

**Decisão (17/11/2026): o protocolo não muda.** As grades continuam as pré-fixadas, e a limitação fica só declarada aqui e no texto. Os motivos: alargar as grades agora, depois de ver quais valores foram escolhidos, seria um ajuste pós-resultado; exigiria refazer a etapa 08 inteira; e a conclusão central não depende disso: o `D_FA` não é significativamente melhor que nenhum clássico no DeLong com Holm (menor p ajustado 0,12), e na SZ a LR espectral já o iguala (AUC 0,852). Quem for ler a tabela de comparação deve ter em mente que os clássicos, sobretudo os temporais, podem estar um pouco abaixo do que conseguiriam com grades mais largas.

## Testes

Testes em `tests/e08_classicos/`:
- **características:** dimensões, ordem, valores contra numpy e scipy, senoide conhecida, sinal constante, valores inválidos, correspondência com a STFT do DTSF, alinhamento do cache;
- **treino:** combinado = temporais e espectrais concatenadas; 150 janelas por pessoa só no ajuste; pesos por pessoa e classe chegando aos três classificadores em todos os ajustes; janelas de previsão independentes do sorteio; cada modelo nas duas tarefas, normalização só nas janelas de ajuste, partes internas disjuntas e com as duas classes, seleção que não muda quando se trocam as características e os rótulos do teste, agregação pela média das janelas sem overlap, reprodutibilidade, saídas separadas das históricas;
- **avaliação:** comparação pareada (`comparacao_pareada`: Δ, IC e Δ por repetição), resumo gravado só na pasta dos clássicos;
- **integridade:** o código compartilhado (`pipeline/avaliacao`, `pipeline/e06_LR_psd`, `pipeline/e05_folds`) inalterado.

## Resultados

Os 9 baselines estão completos nos dois datasets (50 tarefas por modelo e dataset, semente 20261108) e foram avaliados contra o `D_FA` da rodada oficial em GPU. O resumo é [outputs/10_resultados/resumos/08_classicos.md](../outputs/10_resultados/resumos/08_classicos.md). Melhores baselines por AUC por pessoa:

| Dataset | `D_FA` | LR combinado | LR espectral | RF combinado | KNN combinado |
| --- | --- | --- | --- | --- | --- |
| SZ | 0,852 | 0,811 | 0,852 | 0,694 | 0,709 |
| MDD | 0,964 | 0,939 | 0,934 | 0,903 | 0,912 |

- Em SZ, a LR espectral empata com o `D_FA` (ΔAUC +0,000 [−0,061, +0,061]); com as características só temporais, os baselines ficam perto do acaso (AUC 0,56–0,60) e o `D_FA` ganha por +0,25 a +0,29 com IC acima de zero.
- No DeLong com Holm, o `D_FA` não é significativamente melhor que LR, RF ou KNN combinados em nenhum dataset (menor p Holm: 0,12 em MDD contra o RF).

Cada `.json` grava, além dos hiperparâmetros escolhidos, o modelo, a semente, a grade de candidatos, o número de árvores, as partes internas e as janelas por pessoa. Os `.json` de 9/11 não têm os campos novos.

