.PHONY : help
help :
	@echo "clean       	: Remove auto-generated files."
	@echo "test		: Run tests."
	@echo ""
	@echo "All the above targets should be run in pipenv shell."

test:
	rm -f *.pkl
	python3 -m pytest
	rm -f *.pkl

clean:
	rm -f *.pkl
	rm -f -r __pycache__