# Baselines clássicos por modalidade — resultados

Nove baselines clássicos para comparar com o DTSF adaptado: regressão logística, Random Forest e KNN × características temporais, espectrais e combinadas (as duas concatenadas), extraídas das mesmas entradas do DTSF. O plano e a hipótese H4 estão em [00_proposta-tcc.md](00_proposta-tcc.md): a H4 compara o D_FA só com os 3 baselines combinados, e os 6 de uma modalidade só são descritivos.

Os resultados ficam em [Resultados](#resultados).

- Código: [pipeline/e08_classicos/](../pipeline/e08_classicos/)
  - `caracteristicas.py`: 475 características temporais e 1.102 espectrais por janela;
  - `treino.py`: os 9 modelos, as grades e a seleção interna por pessoa;
  - `rodar.py`: o runner, com retomada, processos em paralelo e permutações; monta o combinado concatenando as 475 temporais e as 1.102 espectrais de cada janela (1.577);
  - `avaliar.py`: as AUCs, as comparações D_FA − baseline (o modelo principal da etapa 07), a H4 sobre os 3 combinados (`BASELINES_H4`) e as comparações descritivas com os 6 de uma modalidade.
- Testes: `tests/e08_classicos/`.
- Saídas:
  - `outputs/08_classicos/caracteristicas/`: o cache, com `temporais.npy`, `espectrais.npy` e `extracao.json` (todo o `outputs/` fica fora do git);
  - `outputs/08_classicos/resultados/<dataset>/<modelo>/r<rep>_f<fold>.{csv,json}` e `resumo.md`, que traz a AUC por pessoa com IC e uma tabela com a média ± desvio, nas 10 repetições, da AUC e das secundárias ([avaliacao.md](avaliacao.md#média--desvio-nas-repetições));
  - `outputs/08_classicos/permutacoes/`, se a rodada de permutações for feita (`rodar --permutacoes 25`). **Nenhum código avalia essas permutações ainda**: rodá-las (2.250 treinos) só faz sentido se a avaliação for implementada; a decisão está nas pendências da proposta.
  - Nada é gravado em `outputs/07_DTSF/resultados`, `outputs/07_DTSF/permutacoes` ou `outputs/06_LR_psd`.

Pré-requisitos: as etapas 03 e 05 e os caches da 07 (`pipeline.e07_DTSF.dados` e `pipeline.e07_DTSF.dados_dtsf`). O `avaliar` também precisa das previsões completas do modelo principal (D_FA) em `outputs/07_DTSF/resultados`; sem elas, ele não dá erro, só marca o D_FA como incompleto e não calcula a H4.

```bash
venv/bin/python -m pipeline.e08_classicos.caracteristicas                     # cache das características, uma vez (~1 min)
venv/bin/python -m pipeline.e08_classicos.rodar 2>&1 | tee -a outputs/08_classicos/log_classicos.txt            # 900 tarefas (9 modelos × 2 datasets × 5 folds × 10 repetições); retoma
venv/bin/python -m pipeline.e08_classicos.avaliar 2>&1 | tee -a outputs/08_classicos/log_classicos_avaliar.txt   # resumo e H4 → outputs/08_classicos/resultados/resumo.md
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

12 tarefas (6 modelos × 2 datasets), permutação 0, fold 0, **rótulos embaralhados**, com a saída na pasta temporária da sessão, fora de `outputs/`. Ela confere o formato e o fluxo, e não é resultado.
- As previsões cobrem exatamente as pessoas de teste do fold.
- As partes internas usam só as pessoas de treino e validação.
- Cada pessoa de ajuste entrou com 60 janelas: 1.320 janelas de ajuste na SZ e 2.700 no MDD. Era a versão anterior; com 150, são ~3.300 e ~6.400.
- A tarefa mais cara foi o RF_espectral no MDD, com 160 s. O conjunto levou 2 min 46 s com 10 processos, ~530 s de CPU.
- `avaliar` rodado sobre essa pasta marca todos os baselines como incompletos, não calcula a H4 e grava o `resumo.md` só na pasta indicada.

## Revisão final

Feita sem rótulos reais e sem resultados dos baselines.
- **H4:** ficou documentado no plano, na proposta e no `resumo.md` que a interseção-união e o Holm controlam só a multiplicidade. A validade de cada permutação não é garantida, por causa da dependência da validação cruzada.
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

## Testes

70 testes em `tests/e08_classicos/` (206 no repositório inteiro depois que `outputs/` deixou de ser versionado, H1–H3 ganharam testes e o DeLong e a ablação de 1–40 Hz saíram):
- **características:** dimensões, ordem, valores contra numpy e scipy, senoide conhecida, sinal constante, valores inválidos, correspondência com a STFT do DTSF, alinhamento do cache;
- **treino:** combinado = temporais e espectrais concatenadas; 150 janelas por pessoa só no ajuste; pesos por pessoa e classe chegando aos três classificadores em todos os ajustes; janelas de previsão independentes do sorteio; cada modelo nas duas tarefas, normalização só nas janelas de ajuste, partes internas disjuntas e com as duas classes, seleção que não muda quando se trocam as características e os rótulos do teste, agregação pela média das janelas sem overlap, reprodutibilidade, saídas separadas das históricas;
- **avaliação:** interseção-união, Holm, comparação pareada com ranks (`comparacao_pareada`), resumo gravado só na pasta dos clássicos;
- **integridade:** o código compartilhado (`pipeline/avaliacao`, `pipeline/e06_LR_psd`, `pipeline/e05_folds`) inalterado.

## Resultados

Pendentes da rodada com a semente 20261108.

