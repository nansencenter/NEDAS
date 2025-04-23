Build the docker image `docker build -t qgmodel-benchmark .`

Run a container, mounting your directory to /work inside the container
If you want to keep the generated file with your user id, pass `--build-arg USER_ID=`

`docker run -it --rm -v [path-to-NEDAS]/examples/qgmodel_benchmark:/work qgmodel-benchmark`

In the container, run

Enter the python environment, `. /app/python.src`


