.PHONY : help, checkmicc, clean, lint, test, gendocs
help :
	@echo "checkmicc 	: Check maintainability-index and cyclomatic-complexity."
	@echo "clean       	: Remove auto-generated files."
	@echo "lint     : Tun pylint"
	@echo "test		: Run tests."
	@echo "gendocs		: Generate documentation."
	@echo "opendocs	: Generate and open documentation in default browser."

checkmicc:
	python3 -m radon mi pykitml
	python3 -m radon cc pykitml

clean:
	rm -f *.pkl
	rm -f -r .pytest_cache
	rm -f -r pykitml/__pycache__
	make -C tests/ clean
	make -C docs/ clean
	rm -f -r build/
	rm -f -r dist/

lint:
	pylint pykitml tests  --rcfile ./.pylintrc

test:
	make -C tests/ test

gendocs:
	make -C docs/ clean
	make -C docs/ html

opendocs: gendocs
	xdg-open docs/_build/html/index.html




