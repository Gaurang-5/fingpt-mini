.PHONY: data train generate experiments visualize clean

data:
	python src/data_cleaner.py

train:
	python src/train.py --corpus data/cleaned_corpus.txt --char-vocab data/char_vocab.json --epochs 10

generate:
	python src/generate.py

experiments:
	python src/experiment.py

visualize:
	python src/visualize_attention.py

clean:
	rm -rf checkpoints/*
	rm -rf results/*
