all:
	python3 analise.py
	python3 simulador.py

dem:
	python3 demulador.py

sim:
	python3 simulador.py

gen:
	python3 gerarSimb.py -n 1000 -o simbolos.csv
	python3 simulador.py

clean:
	rm simbolos.csv
