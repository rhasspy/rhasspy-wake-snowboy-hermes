SHELL := bash
PYTHON_NAME = rhasspywake_snowboy_hermes
PACKAGE_NAME = rhasspy-wake-snowboy-hermes
SOURCE = $(PYTHON_NAME)
PYTHON_FILES = $(SOURCE)/*.py bin/*.py *.py
SHELL_FILES = bin/$(PACKAGE_NAME) debian/bin/* *.sh
PIP_INSTALL ?= install
DOWNLOAD_DIR = download

.PHONY: reformat check dist venv test pyinstaller debian docker deploy downloads

version := $(shell cat VERSION)
architecture := $(shell bash architecture.sh)

# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------

reformat:
	scripts/format-code.sh $(PYTHON_FILES)

check:
	scripts/check-code.sh $(PYTHON_FILES)

venv: downloads
	scripts/create-venv.sh

dist: sdist debian

sdist:
	python3 setup.py sdist

test:
	echo "Skipping tests for now"

test-wavs:
	bash etc/test/test_wavs.sh

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------

docker: pyinstaller
	docker build . -t "rhasspy/$(PACKAGE_NAME):$(version)" -t "rhasspy/$(PACKAGE_NAME):latest"

deploy:
	echo "$$DOCKER_PASSWORD" | docker login -u "$$DOCKER_USERNAME" --password-stdin
	docker push "rhasspy/$(PACKAGE_NAME):$(version)"

# -----------------------------------------------------------------------------
# Debian
# -----------------------------------------------------------------------------

pyinstaller: downloads
	scripts/build-pyinstaller.sh "${architecture}" "${version}"

debian: downloads
	scripts/build-debian.sh "${architecture}" "${version}"

# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------

downloads: $(DOWNLOAD_DIR)/snowboy-1.3.0.tar.gz

$(DOWNLOAD_DIR)/snowboy-1.3.0.tar.gz:
	mkdir -p "$(DOWNLOAD_DIR)"
	curl -sSfL -o $@ 'https://github.com/Kitt-AI/snowboy/archive/v1.3.0.tar.gz'
