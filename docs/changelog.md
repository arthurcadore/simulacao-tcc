# CHANGELOG / ROADMAP

---
### v1.0.1 - Base
- [X] Entregar funções base de simulação
    - [X] Montagem de datagrama
    - [X] Codificação convolucional
    - [X] Embaralhamento
    - [X] Codificação de linha
    - [X] Multiplexação
    - [X] Palavra de sincronismo 
    - [X] Formatação de sinal
    - [X] Modulação
    - [X] Adição de ruido 
- [X] Entregar funções base de recepção
    - [X] Demodulação
    - [X] Filtragem passa baixa
    - [X] FIltragem casada
    - [X] Decisão de bit
    - [X] Decodificação de linha
    - [X] Decodificação convolucional
    - [X] Desembaralhamento
    - [X] Desmontagem de datagrama
- [X] BER vs SNR
    - [X] Simulação BER SNR simples via repetição
    - [X] Comparação com QPSK

---
### v1.0.2 - Carrier detection
- [X] Implemetar detecção de portadora
    - [X] adição de portadora no formatador (de acordo com o canal)
    - [X] calculo de segmentação e fft do sinal recebido (portadora pura)
    - [X] verificação de portadora com base em limiar de potência
    - [X] instanciar cadeia de recepção para cada portadora

---
### v1.0.3 - BERSNR e Sincronização
- [X] Alterar escala y do detector para até -40dB
- [X] Adicionar opcionalmente um vetor de payload para o datagrama argos.
- [X] Adicionar curva QPSK teórico (q sqrt(2ebn0))
- [X] Alterar calculo de SNR vs BER para considerar erro de bit
    - [X] Fixar a seed do gerador de números aleatórios para classe noise.
    - [X] Fixar a seed no gerador de números aleatórios para classe datagrama.
- [X] Adicionar sincronismo perfeito
    - [X] Criar vetor de sincronismo com mesmo fs e sequencia esperada
    - [X] Criar função de correlação entre sinal recebido e vetor de sincronismo
    - [X] Identificar instante de maior correlação, instante otimo para amostragem. 
    - [X] Alterar cadeia do receiver para comportar o uso do sincronismo

---
### v1.0.4 - Channel Encoding 
- [X] Alterar codificação de linha pra ficar apenas em NRZ
- [X] Alterar pulso formatador NRZ para ser soma de 2x RRC deslocados
- [X] Aplicar pulso formatador RRC no canal I e pulso MAN no canal Q
- [X] Otimizar MatchedFilter para consumir formatter.
- [X] Otimizar Encoder para ter apenas NRZ e Manchester

---
### v1.0.5 - Optimization
- [X] Otimizar parse do datagrama ARGOS-3
- [X] Remover vetor de tempo no rx, criar interno.
- [X] Otimizar classes de tx/rx, instânciar direto no construtor.
- [X] Alterar detector para checkfrequencies ser chamado dentro da classe
- [X] Adicionar método update para alterar delay do sinal recebido
- [X] Diminuir passo de frequência aleatória na classe detector. 
- [X] Adicionar defaults na construção de classes (argumentos que forem possiveis).
- [] Adicionar truncador no tail do parse do datagrama, pra pegar apenas a cauda e verificar. 
- [] Adicionar método no transmissor para mandar mais sinal após a cauda para testar a truncagem.
- [] Adicionar documentação interna as classes
    - [] Alterar "exemplo" para "example" nos comentários. 
    - [] Alterar API.md para retirar titulo principal e comentar header dos arquivos .py
    - [] Verificar equacionamento sklar e alterar nome das variáveis.
    - [] Alterar documentação da classe de encoder
    - [] adicionar prints de impulso casado na documentação
- [] Otimizar plots
    - [] Verificar plot de constelação, normalização do sinal recebido
    - [] Otimizar plot do encoder. 
    - [] Criar método para testar todas as classes no makefile. 
    - [] Padronizar cor dos plots.
    - [] Alterar tamanho do plot pz fpb
    - [] Otimizar conversor de plots pra permitir darkTheme
    - [] Alterar xlim para plots necessários. 
- [] Adicionar melhor documentação a aba inicio
- [] Adicionar exemplificação de uso como "Example" no site, como o datagrama.
- [] Verificar adição de diagrama de olho no sklar. 
- [] Diagrama de blocos
    - [] Alterar diagrama do modulador
    - [] Alterar nome das variáveis
    - [] Alterar bloco do receptor pra ficar dentro de detector.

---
### v1.0.6 - Channel and Doppler
- [] Montar classe de canal para instanciar ruido e aplicar ao sinal recebido
- [] Montar classe Doppler para testar distorção em frequência do sinal
    - [] Coletar dataset de passada de satélite e montar plot. 
    - [] calcular com base na velocidade de trajetória o desvio doppler. 
    - [] Aplicar distorção doppler no sinal recebido
        - [] Tempo de estabilização 80ms
        - [] Desvio de fase <= 20° no máximo. 
        - [] Fator de amortecimento e=1 
        - [] Máximo desvio de frequência inicial 50Hz
- [] Montar classe PLL (Phased Locked Loop) para corrigir o desvio de frequência do sinal recebido
    - [] Testar aplicação do PLL ao sinal distorcido em frequência.