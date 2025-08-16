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
	.venv/bin/python -m pdf2svg
	@echo "Imagens geradas!"

doc: 
	@echo "Gerando documentação..."
	.venv/bin/mkdocs build
	.venv/bin/mkdocs serve -a 0.0.0.0:8000
		
freeze: 
	@echo "Congelando dependências..."
	.venv/bin/pip freeze > requirements.txt
	@echo "Dependências congeladas em requirements.txt"

deploy-docs: doc
	@echo "Enviando documentação para o GitHub Pages..."
	@if ! git diff --quiet || ! git diff --cached --quiet; then \
		echo "Erro: Existem alterações não commitadas. Por favor, faça commit das alterações antes de fazer o deploy."; \
		exit 1; \
	fi
	@echo "Construindo documentação..."
	@.venv/bin/mkdocs build --site-dir site
	@echo "Corrigindo caminhos das imagens..."
	@find site -type f -name '*.html' -exec sed -i 's|assets/|api/assets/|g' {} \;
	@echo "Fazendo push para o branch gh-pages..."
	@if ! git show-ref --verify --quiet refs/heads/gh-pages; then \
		echo "Criando branch gh-pages..."; \
		git checkout --orphan gh-pages; \
		git reset --hard; \
		echo "Página inicial do GitHub Pages" > index.html; \
		git add index.html; \
		git commit -m "Initial gh-pages commit"; \
		git push -u origin gh-pages; \
	fi
	@git checkout gh-pages
	@git rm -rf --ignore-unmatch *
	@cp -r site/. .
	@git add .
	@git commit -m "Atualização da documentação"
	@git push origin gh-pages
	@git checkout main
	@echo "Documentação publicada com sucesso! Acesse: https://arthurcadore.github.io/simulacao-tcc/"
