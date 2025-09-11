# CHANGELOG/ROADMAP

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
### v1.0.3 - Channel Encoding 
    - [X] Alterar escala y do detector para até -40dB
    - [] Adicionar opcionalmente um vetor de payload para o datagrama argos.
    - [] Adicionar curva QPSK teórico (q sqrt(2ebn0))
- [X] Alterar calculo de SNR vs BER para considerar erro de bit
    - [X] Fixar a seed do gerador de números aleatórios para classe noise.
    - [X] Fixar a seed no gerador de números aleatórios para classe datagrama.
- [ ] Alterar codificação de linha pra ficar apenas em NRZ (sem duplicidade)
- [ ] Alterar pulso formatador NRZ para ser soma de 2x RRC (um invertido)
- [ ] Aplicar pulso formatador RRC do NRZ e pulso formatador MAN do Manchester
- [ ] Otimizar MatchedFilter para consumir formatter.

---
### v1.0.4 - Sincronismo perfeito
- [ ] Adicionar sincronismo perfeito
    - [ ] Criar vetor de sincronismo com mesmo fs e sequencia esperada
    - [ ] Criar função de correlação entre sinal recebido e vetor de sincronismo
    - [ ] Identificar instante de maior correlação, instante otimo para amostragem. 
    - [ ] Alterar cadeia do receiver para comportar o uso do sincronismo

---
### v1.0.5 - Documentation
- [ ] Otimizar classes de plots
    - [ ] Verificar plot de constelação, normalização do sinal recebido
- [ ] Adicionar documentação interna as classes
    - [ ] Verificar equacionamento sklar e alterar nome das variáveis.
    - [ ] Otimizar parse do datagrama ARGOS-3
    - [ ] Adicionar melhor documentação a aba inicio