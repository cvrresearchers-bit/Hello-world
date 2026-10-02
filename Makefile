.PHONY: all test smoke install
install:
	python3 -m pip install -r requirements.txt
all:
	python3 Main.py
smoke:
	NAA_PROFILE=smoke python3 Main.py
test:
	python3 -m pytest -q tests
