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
- [X] Adicionar documentação interna as classes
    - [X] Alterar "exemplo" para "example" nos comentários. 
    - [X] Alterar API.md para retirar titulo principal e comentar header dos arquivos .py
    - [X] Alterar documentação da classe de encoder
    - [X] adicionar prints de impulso casado na documentação
- [X] Adicionar argumentos extras de tipo de codificação / canal no construtor tx/rx
- [X] Inverter a ordem do plot de impulso
- [X] Manter o valor de 80ms de portadora, transferir responsabilidade de variação para o canal.
- [X] Verificar normalização em amplitude para impulso Man
- [X] Alterar ylim nos plots de freq para -60dB no lpf por exemplo. 
- [X] Implementar uso da função Q(x) na documentação. 
- [X] Calcular o tamanho do numblocks com base no classmethod (mais de um construtor)

--- 
### v1.0.6 - Multiple Carrier detection
- [X] Adicionar truncador no tail do parse do datagrama, pra pegar apenas a cauda e verificar. 
- [X] Alterar classe detector para permitir detecção fora de fase 
    - [X] Criar matriz de frequências vs segmentos no construtor usando deltaF
    - [X] Segmentar o sinal recebido inteiro e alimentar a matriz. 
    - [X] Aplicar FFT em cada bloco, e verificar a potência em cada frequência, gerar a matriz de potência
    - [X] Identificar as frequências confirmadas usando histórico de segmento 
    - [X] Decidir quais frequências serão usadas para demodulação, e criar span para proteção do sinal
    - [X] Demodular o sinal recebido para a sequência de segmentos detectada
- [X] Adicionar método de canal para adicionar ruido dinamicamente
    - [X] Adicionar multiplicador de comprimento do sinal TX
    - [X] Criar vetor de ruido com comprimento maior 
    - [X] Somar o sinal com ruido, colocar argumento de deslocamento (amostra onde começa a soma). 
- [X] Montar diagrama de waterfall
    - [X] criar matriz (frequencia x segmento) 2D
    - [X] criar matriz (frequencia x segmento x potência) 3D
    - [X] Plotar uma cor para cada frequência em cada segmento de tempo para decidido
        - [X] não detectada
        - [X] detectada
        - [X] penalizada 
        - [X] iniciando demodulação

---
### v1.0.7 - Documentation and Styling
- [X] Otimizar conversor de plots pra permitir darkTheme
- [X] Alterar mkdocs para darkTheme
- [X] Adicionar melhor documentação a aba inicio
- [X] Adicionar exemplificação de uso como "Example" no site, como o datagrama.
- [X] Verificar equacionamento sklar e alterar nome das variáveis.
- [X] Diagrama de blocos
    - [X] Alterar diagrama do modulador
    - [X] Alterar nome das variáveis
    - [X] Alterar bloco do receptor pra ficar dentro de detector.
    - [X] Alterar indices no diagrama para ficar no mesmo tamanho. 
    - [X] waterfall, transpor 
- [X] Otimizar plots
    - [X] Verificar plot de constelação, normalização do sinal recebido
    - [X] Otimizar plot do encoder. 
    - [X] Criar método para testar todas as classes no makefile. 
    - [X] Alterar tamanho do plot pz fpb
    - [X] Alterar xlim para plots necessários. 
    - [X] Adicionar plot de fase discreta no transmitter, deixar constelação separado. 
    - [X] Padronizar cor dos plots.
- [X] Criar diretório para armazenar diagramas e notebooks.    

--- 
### v1.0.8 - Soft Decision and Performance
- [] Implementar argumento "soft" no receiver
    - [] Alterar receiver para apenas amostrar os valores após filtro casado.
    - [] Alterar ordem do conv. decoder para receber valores após desembaralhamento.
    - [] Alterar algorimo de viterbi para trabalhar com distâncias euclidianas (ref: https://dsplog.com/2009/01/14/soft-viterbi/)
    - [] Verificar viabilidade de plot de módulo de treliça para plot de estados. 
    - [] Verificar viabilidade de plot com x=tempo e y=estados.
- [] Aplicar novos gráficos de BERvs EBN0
    - [] QPSK usando modulator
    - [] QPSK usando modulator + formatter
    - [] QPSK usando modulator + formatter + hard decision
    - [] QPSK usando modulador + formatter + soft decision 
    - [] QPSK usando modulador + formatter + soft decision + scrambler

---
### v1.0.9 - Channel and Doppler
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


---
### v1.0.10 - AGC and Preamble
- [] Adicionar palavra de sincronismo direto no formatter, remover multiplexador.
    - [] Adicionar vetor de simbolos de preâmbulo 
    - [] Aproveitar esse novo vetor no synchornizer pra fazer a filtragem casada e correlação.



