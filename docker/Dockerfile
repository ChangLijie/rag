FROM  pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

WORKDIR /workspace

COPY ./docker /workspace/docker
COPY ["./docker/entrypoint", "/usr/bin/" ]

RUN chmod u+x /workspace/docker/requirements.sh \
&& /workspace/docker/requirements.sh

ENTRYPOINT [ "entrypoint" ]