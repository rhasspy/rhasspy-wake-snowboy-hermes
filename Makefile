SHELL := bash
PYTHON_FILES = rhasspywake_snowboy_hermes/*.py setup.py

.PHONY: check dist venv test pyinstaller debian

version := $(shell cat VERSION)
architecture := $(shell dpkg-architecture | grep DEB_BUILD_ARCH= | sed 's/[^=]\+=//')

debian_package := rhasspy-wake-snowboy-hermes_$(version)_$(architecture)
debian_dir := debian/$(debian_package)

check:
	flake8 --exclude=snowboy.py $(PYTHON_FILES)
	pylint --ignore=snowboy.py $(PYTHON_FILES)
	mypy $(PYTHON_FILES)
	isort $(PYTHON_FILES)
	black .
	pip list --outdated

snowboy-1.3.0.tar.gz:
	curl -sSfL -o $@ 'https://github.com/Kitt-AI/snowboy/archive/v1.3.0.tar.gz'

venv: snowboy-1.3.0.tar.gz
	rm -rf .venv/
	python3 -m venv .venv
	.venv/bin/pip3 install wheel setuptools
	.venv/bin/pip3 install -r requirements.txt
	.venv/bin/pip3 install $<
	.venv/bin/pip3 install -r requirements_dev.txt

dist: sdist debian

sdist:
	python3 setup.py sdist

test:
	bash etc/test/test_wavs.sh

pyinstaller:
	mkdir -p dist
	pyinstaller -y --workpath pyinstaller/build --distpath pyinstaller/dist rhasspywake_snowboy_hermes.spec
	tar -C pyinstaller/dist -czf dist/rhasspy-wake-snowboy-hermes_$(version)_$(architecture).tar.gz rhasspywake_snowboy_hermes/

debian: pyinstaller
	mkdir -p dist
	rm -rf "$(debian_dir)"
	mkdir -p "$(debian_dir)/DEBIAN" "$(debian_dir)/usr/bin" "$(debian_dir)/usr/lib"
	cat debian/DEBIAN/control | version=$(version) architecture=$(architecture) envsubst > "$(debian_dir)/DEBIAN/control"
	cp debian/bin/* "$(debian_dir)/usr/bin/"
	cp -R pyinstaller/dist/rhasspywake_snowboy_hermes "$(debian_dir)/usr/lib/"
	cd debian/ && fakeroot dpkg --build "$(debian_package)"
	mv "debian/$(debian_package).deb" dist/

docker: pyinstaller
	docker build . -t "rhasspy/rhasspy-wake-snowboy-hermes:$(version)"
