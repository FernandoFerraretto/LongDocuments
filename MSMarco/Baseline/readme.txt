https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz

ref: https://github.com/microsoft/MSMARCO-Passage-Ranking
relacionar queries a reposts

- datast original: triples.train.small.tsv
- 40M triplas query-passage positiva e passagem negativa

---------------  V01 - criado em 22/Ago/2021
- dataset augmentation: msmarco_noise
    - dado uma tripla: QUERY, TXT_POS, TXT_NEG
    - a querya recebeu como ruido o texto, no seguinte esquema:
          amostra pares 0,2,4,N: QUERY + NOISE(TXT_NEG), TXT_POS -> 1
        amostra impares 1,3,5,N: QUERY + NOISE(TXT_POS), TXT_NEG -> 0
        onde, NOISE é o texto dividio em 2 partes iguais
    - utilizou-se as tags TEXT OFF e TEXT ON para indicar o trecho de ruido e texto relevante
    - stats:
        # 400,000 amostras
        # title (seq1)
            max: 363 tokens
            avg:  96 tokens
        # text (seq2)
            max: 396 tokens
            avg:  82 tokens
        # seq1 + seq2
            max: 534 tokens
            avg: 178 tokens

---------------  V02 - criado em 29/Ago/2021
- dataset augmentation: msmarco_noise
    - Dada uma tripla: QUERY | POS | NEG
    - Encontra mais 2 triplas com a mesma query, e selecionar a sequência NEG
    - Assim teremos: QUERY | POS | NEG | NEG1 | NEG2
    - Montar o dataset como:
        text off: NEG/2  text on: QUERY text off: NEG/2  | text on:  POS   | 1
        text off: NEG2/2 text on: QUERY text off: NEG2/2 | text off: NEG1  | 0
    - stats:
        # 22,396 amostras
        # seq1 + seq2
            range: 50-467 tokens
              avg:    176 tokens
- dataset augmentation: msmarco_longnoise250
    - idem ao msmarco_noise
    - repetir o ruido em trechos aleatorios até somar 250 em cada parte (inicio e fim)
    - Montar o dataset como:
        text off: NEG/2-LOOP  text on: QUERY text off: NEG/2-LOOP  | text on:  POS   | 1
        text off: NEG2/2-LOOP text on: QUERY text off: NEG2/2-LOOP | text off: NEG1  | 0
    - stats:
        # 22,396 amostras
        # seq1 + seq2
            range: 546-11865 tokens
              avg:       806 tokens
- dataset augmentation: msmarco_longnoise450
    - idem ao msmarco_noise
    - repetir o ruido em trechos aleatorios até somar 450 em cada parte (inicio e fim)
    - Montar o dataset como:
        text off: NEG/2-LOOP  text on: QUERY text off: NEG/2-LOOP  | text on:  POS   | 1
        text off: NEG2/2-LOOP text on: QUERY text off: NEG2/2-LOOP | text off: NEG1  | 0
    - stats:
        # 22,396 amostras
        # seq1 + seq2
            range: 950-21550 tokens
              avg:      1337 tokens

---------------  V03 - criado em 01/Set/2021
- dataset augmentation: msmarco_longnoise250
    - idem ao msmarco_longnoise250 V02
    - repetir o ruido por completo até somar pelo menos 250 tks em cada parte (inicio e fim)
    - Montar o dataset como:
        text off: NEG/2-LOOP  text on: QUERY text off: NEG/2-LOOP  | text on:  POS   | 1
        text off: NEG2/2-LOOP text on: QUERY text off: NEG2/2-LOOP | text off: NEG1  | 0
    - stats:
        # 22,396 amostras
        # seq1 + seq2
            range: 548-17628 tokens
              avg:       844 tokens



    