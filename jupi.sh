docker run --name jupyter -it -p 8888:8888 \
  -u 1000 -v /Users/agustinfusaro/uavia/gen_path_finder:/tmp/notebooks \
  -e HOME=/tmp/jupyter python:3.8 bash -c "
    mkdir /tmp/jupyter; \
    pip install --user 'jupyterlab < 4' 'ipympl < 0.8' pandas matplotlib gaft; \
    /tmp/jupyter/.local/bin/jupyter lab --ip=0.0.0.0 --port 8888 \
      --no-browser --notebook-dir /tmp/notebooks;
  "
