FROM ubuntu:jammy

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install build-essential clang-15 golang cmake ninja-build \
        python3-venv scons curl git vim && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -rm -s /bin/bash fhelipe
USER fhelipe
WORKDIR /home/fhelipe/

RUN python3 -m venv fhenv
ENV PATH="/home/fhelipe/fhenv/bin:$PATH"

COPY --chown=fhelipe:fhelipe . ./fhelipe

WORKDIR fhelipe
RUN pip install -e frontend/ && pip cache purge

WORKDIR backend
RUN scons lib
RUN scons deps --no-deps-pull
RUN scons -j16 --release

USER fhelipe
WORKDIR ..
