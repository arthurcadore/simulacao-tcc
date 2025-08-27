# Guia de Instalação pip

## Requisitos

- `Python 3.8` (ou superior);
- `Python-venv 3.12.7` (ou superior);
- `pip 24.0` (ou superior);

## Instalação

Para saber as versões disponiveis para instalação, e seu changelog, verifique as releases no [Repositório](https://github.com/arthurcadore/simulacao-tcc/releases).

### Criar ambiente virtual

Inicialmente, crie um ambiente virtual para isolar as dependências do projeto.

```bash
python -m venv .venv
```

Ative o ambiente virtual:

```bash
source .venv/bin/activate
```

### Ultima versão

Para instalar a ultima versão, execute:

```bash
pip install argos3
```

### Versão específica

Para instalar uma versão específica, execute o comando abaixo, substituindo `1.0.1` pela versão desejada: 

```bash
pip install argos3==1.0.1
```

## Exemplo de Uso

Após a instalação da biblioteca, verifique a documentação de classes para mais detalhes, abaixo está um exemplo de uso:

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