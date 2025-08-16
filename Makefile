.PHONY: all install test export-pdf export-pdf-show export-all clean

all: install 

install: 
	@echo "Verificando Python..."
	@python3 --version || (echo "Python não encontrado. Por favor, instale o Python primeiro." && exit 1)
	@echo "Criando ambiente virtual..."
	python3 -m venv .venv
	@echo "Instalando dependências..."
	.venv/bin/pip install -r requirements.txt
	@echo "Instalação concluída!"

install-additional:
	@sudo apt update
	@sudo apt install texlive-latex-extra texlive-fonts-recommended dvipng cm-super -y

clean:
	rm -rf .venv
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

doc-images:
	@echo "Gerando imagens para a documentação..."
	.venv/bin/python docs/pdf2svg.py
	@echo "Imagens geradas!"

doc: 
	@echo "Gerando documentação..."
	mkdocs build
	mkdocs serve -a 0.0.0.0:8001
		
freeze: 
	@echo "Congelando dependências..."
	.venv/bin/pip freeze > requirements.txt
	@echo "Dependências congeladas em requirements.txt"

deploy-docs:
	@echo "Gerando documentação..."
	@if ! command -v mkdocs &> /dev/null; then \
		echo "MkDocs não encontrado. Instalando..."; \
		.venv/bin/pip install mkdocs mkdocs-material; \
	fi
	@echo "Fazendo deploy para o GitHub Pages..."
	@mkdocs gh-deploy --force
	@echo "Documentação publicada com sucesso no GitHub Pages!"