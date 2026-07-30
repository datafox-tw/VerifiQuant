TEXINPUTS := .:../template:
TECTONIC := tectonic

.PHONY: main full clean-build

main:
	TEXINPUTS="$(TEXINPUTS)" $(TECTONIC) main.tex

full:
	TEXINPUTS="$(TEXINPUTS)" $(TECTONIC) main_with_appendix.tex

clean-build:
	rm -f main.pdf main_with_appendix.pdf main.aux main.bbl main.blg main.log main.out main_with_appendix.aux main_with_appendix.bbl main_with_appendix.blg main_with_appendix.log main_with_appendix.out
