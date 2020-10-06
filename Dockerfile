FROM ubuntu:16.04 

# Install Octave.
RUN apt-get update && \
  apt-get install -y --no-install-recommends locales && \
  locale-gen en_US.UTF-8 && \
  apt-get dist-upgrade -y && \
  apt-get install -y --no-install-recommends octave && \
  apt-get clean all

#RUN mkdir /Result
RUN rm -rf home/* 

ADD . /home
#COPY ./main.sce /main.sce
#COPY ./entrypoint.sh /entrypoint.sh

WORKDIR /home/octave

ENTRYPOINT ["/home/entrypoint.sh"]
#ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
