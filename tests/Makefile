.PHONY : help viewprofiles clean $(wildcard *.dat)
help :
	@echo "clean       	: Remove auto-generated files."
	@echo "test		: Run tests."
	@echo "viewprofiles	: View all *.dat profile files."

test:
	rm -f *.pkl
	python3 -m pytest
	rm -f *.pkl

viewprofiles: $(wildcard *.dat)

$(wildcard *.dat):
	$(eval name=$(patsubst %64,%,$(basename $(notdir $@))))
	gprof2dot -f pstats $@ | dot -Tpng -o profile_$(name).png
	xdg-open profile_$(name).png

clean:
	rm -f *.pkl
	rm -f -r __pycache__
	rm -f *.dat
	rm -f profile_*.png