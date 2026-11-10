# Auditoria das gravações EC

A auditoria fixou o n final em **28 pessoas na task SZ (14 SZ + 14 HC)** e **56 na task MDD (29 MDD + 27 HC)**. Nenhuma gravação caiu por duração, saturação ou erro de leitura. Só caíram as duas cópias de gravações duplicadas.

- Código: [pipeline/e01_auditoria/auditar_ec.py](../pipeline/e01_auditoria/auditar_ec.py)
- Critérios: seção [Critérios](#critérios)
- Saídas: `outputs/01_auditoria/auditoria_ec.csv` (uma linha por arquivo) e `resumo.md`

```bash
venv/bin/python -m pipeline.e01_auditoria.auditar_ec
```

## Critérios

Critérios de inclusão das gravações de olhos fechados (EC). A auditoria define o n final de cada task.

### Escopo

| | Task SZ | Task MDD |
|---|---|---|
| Arquivos EC | 28 (`h01`–`h14`, `s01`–`s14`) | 58 (`H S<n> EC`, `MDD S<n> EC`) |
| Frequência | 250 Hz | 256 Hz |
| Canais usados | os 19 do sistema 10–20 | os 19 do sistema 10–20 (`EEG <canal>-LE`); `A2-A1`, `23A-23R` e `24A-24R` são ignorados |

**Arquivos ausentes no MDD.** O figshare (v2, doi:10.6084/m9.figshare.4244171.v2) lista 64 arquivos EC, mas 6 deles nunca ficaram disponíveis para download: a API mostra esses arquivos sem MD5 calculado e o link de cada um retorna 404. São eles `H S12`, `H S18`, `MDD S4`, `MDD S8`, `MDD S12` e `MDD S16`. Os outros 58 conferem com o MD5 da fonte. O mesmo acontece com 6 arquivos EO/TASK, que estão fora do escopo.

### Unidade de análise

A unidade é a **pessoa**. Cada pessoa recebe um `subject_uid`, e cada pessoa contribui com no máximo uma gravação EC. Todos os splits são feitos por `subject_uid`.

### Decisões possíveis

- **INCLUIR:** a gravação passa em todas as checagens.
- **SINALIZAR:** a gravação entra na análise principal, mas o motivo fica registrado e é reportado.
- **EXCLUIR:** a gravação fica fora, com o motivo registrado.

### Checagens

| # | Checagem | Regra | Decisão |
|---|---|---|---|
| C1 | Leitura e formato | O arquivo abre no MNE, tem a frequência esperada e tem os 19 canais 10–20 | Falhou → EXCLUIR |
| C2 | Arquivo duplicado | Dois arquivos com o mesmo SHA-256 | Mesma pessoa e mesmo rótulo → uma pessoa só, mantém um arquivo e SINALIZA. Rótulos diferentes → EXCLUIR os dois |
| C3 | Sinal duplicado | Dois arquivos com os 19 canais idênticos amostra a amostra (depois de cortar os zeros do fim), mesmo com SHA-256 diferente | Mesma regra de C2 |
| C4 | Identidade no cabeçalho | O campo de paciente do EDF é igual entre arquivos de nomes diferentes, ou indica outra condição (EO/TASK) | SINALIZAR (o cabeçalho pode não ter sido atualizado). Se coincidir com C2/C3, vale C2/C3 |
| C5 | Zeros no fim | Trecho final em que todos os canais valem exatamente 0 | Cortado. Não exclui |
| C6 | Duração útil | Duração depois do corte de C5 | Menos de 122 s → EXCLUIR. 122 s é o mínimo para 60 janelas de 4 s com 50% de overlap, o número que a especificação sorteia por pessoa a cada época |
| C7 | Canal plano | Trecho contínuo de pelo menos 10 s com desvio padrão abaixo de 0,5 µV em janelas de 1 s | Plano em mais de 20% da duração útil → canal ruim. Trechos menores ficam marcados e, se o canal não for interpolado (C9), as janelas que os tocam são descartadas depois, na etapa de janelas |
| C8 | Saturação | Mais de 1% das amostras de um canal no limite digital do EDF | Canal ruim |
| C9 | Canais ruins (C7 + C8) | Contagem por gravação | 1 canal → interpolado por spline esférica e SINALIZAR. 2 ou mais → EXCLUIR |

**Fora da auditoria.** Artefatos de amplitude (o limiar de 150 µV) são tratados janela a janela, depois da harmonização. A auditoria olha a gravação inteira e não descarta janelas.

## Resultados

### n final

| Task | Classe | Arquivos EC | INCLUIR | SINALIZAR | EXCLUIR | Pessoas |
| --- | --- | --- | --- | --- | --- | --- |
| SZ | HC | 14 | 14 | 0 | 0 | 14 |
| SZ | SZ | 14 | 13 | 1 | 0 | 14 |
| MDD | HC | 28 | 25 | 2 | 1 | 27 |
| MDD | MDD | 30 | 28 | 1 | 1 | 29 |

No MDD, a auditoria parte de 58 arquivos EC, e não dos 64 listados no Figshare: 6 nunca ficaram disponíveis para download (ver [Escopo](#escopo)).

### Exclusões e sinalizações

| Arquivo | Decisão | Motivo |
| --- | --- | --- |
| H S30 EC.edf | EXCLUIR | C2: cópia byte a byte de H S27 EC; as duas contam como uma pessoa |
| MDD S34 EC.edf | EXCLUIR | C2: cópia byte a byte de MDD S33 EC; as duas contam como uma pessoa |
| H S27 EC.edf | SINALIZAR | Original da duplicata acima; entra |
| MDD S33 EC.edf | SINALIZAR | Original da duplicata acima; entra |
| H S14 EC.edf | SINALIZAR | C4: o cabeçalho EDF indica a condição EO; entra |
| s14.edf | SINALIZAR | C9: canal C3 plano em 33% da gravação; é interpolado na harmonização |

### Outros achados

- **Zeros no fim:** 27 dos 28 arquivos da SZ (nenhum do MDD) terminam com até 4,7 s de amostras zeradas em todos os canais. O trecho é cortado (C5) e não causa exclusão.
- **Duração útil:** a menor gravação EC do MDD tem 180 s (MDD S3), acima do mínimo de 122 s.
- **Privacidade:** o campo de paciente do EDF do MDD traz nome e data de nascimento. A auditoria guarda só um hash desse campo.
