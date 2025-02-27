# Variables
VENV_NAME = venv
REQUIREMENTS = requirements.txt
PYTHON = python
PIP = pip
LINTER = flake8
FORMATTER = black
SECURITY = bandit

# Etape 1: Création de l'environnement virtuel et installation des dépendances
install: $(VENV_NAME)/bin/activate
	$(PIP) install -r $(REQUIREMENTS)

$(VENV_NAME)/bin/activate:
	python -m venv $(VENV_NAME)
	$(PIP) install --upgrade pip
	touch $(VENV_NAME)/bin/activate

# Etape 2: Vérification du code (Formatage, Qualité, Sécurité)
format:
	$(FORMATTER) .  # Formatte tous les fichiers Python

lint:
	$(LINTER) .  # Vérifie la qualité du code Python

security:
	$(SECURITY) .  # Vérifie les problèmes de sécurité dans le code

# Etape 3: Préparer les données
prepare:
	$(PYTHON) main.py --file_path merged_churn.csv --prepare

# Etape 4: Entraîner le modèle
train:
	$(PYTHON) main.py --file_path merged_churn.csv --train

# Etape 5: Tester le modèle
evaluate:
	$(PYTHON) main.py --file_path merged_churn.csv --evaluate

# Etape 6: Exécuter l'intégralité du pipeline
all: install format lint security prepare train evaluate

# Makefile

# Target to run the FastAPI server
run_api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

run-mlflow:
	mlflow ui --host 0.0.0.0 --port 5000 &
# Test with pytest
test:
	source venv/bin/activate && pytest tests/

