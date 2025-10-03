# Manual Installation

## Requirements

- `Python 3.8` (or higher);
- `Python-venv 3.12.7` (or higher);
- `Git 2.43.0` (or higher);
- `GNU Make 4.3` (or higher);
- `pdfTeX 3.141592653-2.6-1.40.25` (or higher);
- `kpathsea version 6.3.5` (or higher);
- Recomend Ubuntu 24.04.2 LTS for following the installation/update of packages.  

## Installation

### Cloning the repository

First, clone the repository, containing the project: 

```bash
git clone https://github.com/arthurcadore/argos3.git
```

Next, access the project directory: 

```bash
cd simulacao-tcc
```

### Installation with Makefile

In the project directory, update the system packages: 

```bash
apt-get update
```

Install the dependencies necessary for the Makefile:

```bash
apt-get install python3-venv git make -y
```

The project includes a Makefile that automates the creation of the virtual environment and the installation of dependencies listed in `requirements.txt`.

```bash
make install
```

Next, install additional dependencies for PDF generation (Linux) using latex, necessary for the documentation: 

```bash
make install-additional
```

### Activate the virtual environment

After installation, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## Documentation

### Visualizing Locally

To view the documentation locally, use:

```bash
make doc
```

This will start a local server at `http://localhost:8005`

### PDF to SVG Conversion

To generate the images used in the web documentation, convert the PDF files to SVG using the Makefile:

```bash
make doc-images
```

### Deploy Documentation

To deploy the documentation to GitHub Pages, use:

```bash
make deploy-docs
```

## Support

To report issues or request features, please [open an issue](https://github.com/arthurcadore/argos3/issues) on GitHub.

## License

This project is part of the IFSC - Federal Institute of Santa Catarina. Consult the [LICENSE](https://github.com/arthurcadore/argos3/blob/main/LICENSE) file for more details about the terms of use.

## Contact

- **Author**: Arthur Cadore M. Barcella
- **Professors**: 
  - Prof. Roberto Wanderley da Nóbrega, Dr.
  - Prof. Richard Demo Souza, Dr.
- **Institution**: IFSC - Federal Institute of Santa Catarina