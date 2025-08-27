# Instalação Manual

## Requisitos

- `Python 3.8` (ou superior);
- `Python-venv 3.12.7` (ou superior);
- `Git 2.43.0` (ou superior);
- `GNU Make 4.3` (ou superior);
- `pdfTeX 3.141592653-2.6-1.40.25` (ou superior);
- `kpathsea version 6.3.5` (ou superior);
- Recomenda-se a distribuição `Ubuntu 24.04.2 LTS` para seguir os passos de instalação/atualização de pacotes.  

## Instalação

### Clonar o repositório

Inicialmente faça o clone do repositório, contendo o projeto: 

```bash
git clone https://github.com/arthurcadore/simulacao-tcc.git
```

Em seguida, acesse o diretório do projeto: 

```bash
cd simulacao-tcc
```

### Instalação com Makefile

No diretório do projeto, atualize os pacotes do sistema: 

```bash
apt-get update
```

Instale as dependências necessárias para o Makefile:

```bash
apt-get install python3-venv git make -y
```

O projeto inclui um Makefile que automatiza a criação do ambiente virtual e a instalação das dependências listadas no `requirements.txt`.

```bash
make install
```

Em seguida, instale as dependências adicionais para geração de PDF (Linux) utilizando latex, necessário para a documentação: 

```bash
make install-additional
```

### Ativar o ambiente virtual

Após a instalação, ative o ambiente virtual com:

```bash
source .venv/bin/activate
```

## Documentação

### Visualizando Localmente

Para visualizar a documentação localmente, use:

```bash
make doc
```

Isso irá iniciar um servidor local em `http://localhost:8005`

### Conversão PDF para SVG

Para gerar as imagens usadas na documentação web, converta os arquivos PDF para SVG com o Makefile:

```bash
make doc-images
```

### Publicar a Documentação

Para publicar a documentação no GitHub Pages, use:

```bash
make deploy-docs
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