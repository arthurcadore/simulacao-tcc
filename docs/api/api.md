# Guia de Instalação PIP

## Requisitos

- Python 3.8+
- NumPy
- Matplotlib (para visualizações)
- SciPy (para processamento de sinal)

## Instalação

### Instalação da última versão: 

Para instalar a ultima versão, execute:

```bash
pip install argos3
```

### Versão específica:

Para instalar uma versão específica, execute: 

```bash
pip install argos3==1.0.0
```

Para saber as versões disponiveis para instalação, e seu changelog, verifique as releases no [Repositório](https://github.com/arthurcadore/simulacao-tcc/releases).

## Exemplo de Uso:

```python
>>> from argos3 import Datagram
>>> datagram_tx = Datagram(1234, 2)
>>> 
>>> print(datagram_tx.parse_datagram())
{
  "msglength": 2,
  "pcdid": 1234,
  "data": {
    "bloco_1": {
      "sensor_1": 6,
      "sensor_2": 7,
      "sensor_3": 250
    },
    "bloco_2": {
      "sensor_1": 159,
      "sensor_2": 13,
      "sensor_3": 248,
      "sensor_4": 59
    }
  },
  "tail": 8
}
>>> 
```

## Suporte

Para relatar problemas ou solicitar recursos, por favor [abra uma issue](https://github.com/arthurcadore/simulacao-tcc/issues) no GitHub.

## Licença

Este projeto faz parte do Trabalho de Conclusão de Curso em Engenharia de Telecomunicações do IFSC. Consulte o arquivo [LICENSE](https://github.com/arthurcadore/simulacao-tcc/blob/main/LICENSE) para mais detalhes sobre os termos de uso.

## Contato

- **Autor**: Arthur Cadore M. Barcella
- **Orientadores**: 
  - Prof. Roberto Wanderley da Nóbrega, Dr.
  - Prof. Richard Demo Souza, Dr.
- **Instituição**: IFSC - Instituto Federal de Santa Catarina