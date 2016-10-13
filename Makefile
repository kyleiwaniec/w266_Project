SHELL=/bin/bash

images:
	pushd scoringApp && docker build -t w266/scoring-app . && popd

app: images
	docker run -it --rm \
		-v `pwd`/data:/usr/data \
		-p 8085:8085 \
	    w266/scoring-app
