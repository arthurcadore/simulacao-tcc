# Guia de Instalação Manual

Guia de instalação para o Simulador ARGOS-3. 

## Pré-requisitos

- `Python 3.8` (ou superior);
- `Python-venv 3.12.7` (ou superior);
- `Git 2.43.0` (ou superior);
- `GNU Make 4.3` (ou superior);
- Recomenda-se a distribuição `Ubuntu 24.04.2 LTS` para seguir os passos de instalação/atualização de pacotes.  

## Instalação

### 1. Clonar o repositório

Faça o clone do repositório, e acesse o diretório do projeto.

```bash
git clone https://github.com/arthurcadore/simulacao-tcc.git
cd simulacao-tcc
```

### 2. Instalação com Makefile

Inicialmente, atualize os pacotes do sistema. Para sistemas baseados em Debian, como Ubuntu, use:

```bash
apt-get update && apt-get upgrade -y
```

O projeto inclui um Makefile que automatiza a criação do ambiente virtual e a instalação das dependências listadas no `requirements.txt`.

```bash
make install
```

#### 2.1 Instalação Latex

Instala dependências adicionais para geração de PDF (Linux) utilizando latex, necessário para a documentação, para sistemas baseados em Debian, como Ubuntu, use: 

```bash
make install-additional
```

Versões recomendadas: 

- `pdfTeX 3.141592653-2.6-1.40.25 (TeX Live 2023/Debian)`
- `kpathsea version 6.3.5`

### 3. Ativar o ambiente virtual

Após a instalação, ative o ambiente virtual com:

```bash
source .venv/bin/activate
```

## Gerar Documentação

### 1. Visualizando Localmente

Para visualizar a documentação localmente, use:

```bash
make doc
```

Isso irá iniciar um servidor local em `http://localhost:8005`

### 2. Conversão PDF para SVG

Para gerar as imagens usadas na documentação web, converta os arquivos PDF para SVG com o Makefile:

```bash
make doc-images
```

### 3. Publicar a Documentação

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