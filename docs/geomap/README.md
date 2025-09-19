# Geomap

### Fonte: http://sinda.crn.inpe.br/PCD/SITE/novo/site/historico/index.php

---

#### geoencoder.py
Script para montar lista de PCDs e suas respectivas coordenadas geográficas, como entrada deve receber um arquivo CSV com os dados de PCDs: 

```csv
32105,AC,Assis Brasil,
32392,AC,Brasileia,
32076,AC,Cruzeiro do Sul,
32383,AC,Cruzeiro do Sul,
32106,AC,Fazenda Santo Afonso,
32083,AC,Feijo,
32073,AC,Foz do Breu,
32082,AC,Iratapuru,
32100,AC,Jusante Rio Preto,
32101,AC,Manoel Urbano,
32591,AC,MET Cruzeiro do Sul,
31901,AC,PARNA  S. Divisor,
32150,AC,Placido de Castro (7),
32120,AC,Porto Valter,
31909,AC,Rio Branco,
32107,AC,Rio Branco (6),
32099,AC,Rio Cmte. Fontoura,
32077,AC,Seringal Bom Futuro,
```

Para cada PCD será feita uma requisição à API do Nominatim para obter as coordenadas geográficas, resultando em um arquivo CSV com as coordenadas de cada PCD.

```csv
32105,"Assis Brasil, AC",-10.9409203,-69.5672108
32392,"Brasileia, AC",-11.0010413,-68.7487894
32076,"Cruzeiro do Sul, AC",-7.6362478,-72.6691649
32383,"Cruzeiro do Sul, AC",-7.6362478,-72.6691649
32106,"Fazenda Santo Afonso, AC",-30.9487835,-53.6544701
32083,"Feijo, AC",-8.1648652,-70.3539579
32073,"Foz do Breu, AC",-9.410421,-72.715408
32082,"Iratapuru, AC",-0.4988508,-52.5713751
32101,"Manoel Urbano, AC",-8.838755,-69.2619465
32150,"Placido de Castro (7), AC",-10.001534,-67.8405833
32120,"Porto Valter, AC",-8.2687722,-72.7444458
31909,"Rio Branco, AC",-9.9765362,-67.8220778
32107,"Rio Branco (6), AC",-9.9439907,-67.8179262
32077,"Seringal Bom Futuro, AC",-8.7169963,-71.0283084
```

### Gerando o mapa

```bash
make
```

Como saída será gerado um arquivo PDF com o mapa dos PCDs.

![map_pcds](../assets/geoplot.svg)


