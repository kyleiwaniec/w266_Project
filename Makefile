SHELL=/bin/bash

images:
	pushd import-ka-comments && docker build -t w266/import-ka-comments . && popd
	pushd scoringApp && docker build -t w266/scoring-app . && popd
	pushd models && docker build -t w266/scoring-notebook . && popd

importka: images
	mkdir -p ./data
	docker run -it --rm \
		-v `pwd`/data:/usr/data \
		w266/import-ka-comments

app: images
	docker run -it --rm \
		-v `pwd`/data:/usr/data \
		-p 8085:8085 \
	    w266/scoring-app

notebook: images
	docker run -it --rm \
		-v `pwd`:/usr/src/app \
		-p 9126:9123 \
		w266/scoring-notebook
