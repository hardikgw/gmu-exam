#!/usr/bin/env bash
#!/usr/bin/env bash
docker run -it -p 8888:8888 -p 6006:6006 --name=tensorflow -v $(pwd)/notebooks:/notebooks -v $(pwd)/logs:/logs -v $(pwd)/models:/models -e PASSWORD=password cithub/tensorflow