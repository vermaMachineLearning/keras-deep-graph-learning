FROM ubuntu:latest

# Install.
RUN \
  sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && \
  apt-get update && \
  apt-get -y upgrade && \
  apt-get install -y build-essential && \
#  apt-get install default-jre default-jdk maven -y && \
  apt-get install -y software-properties-common && \
  apt-get install -y byobu curl git htop man unzip vim wget zip && \
  apt-get install python3.6 python3-pip -y && \
  rm -rf /var/lib/apt/lists/*

# Add files.

COPY ./ /root/
# Set environment variables.
ENV HOME /root

# Define working directory.
WORKDIR /root

RUN pip3 install keras theano tensorflow 

# Define default command.
CMD ["bash"]

