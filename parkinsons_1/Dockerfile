FROM ubuntu

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install Flask

WORKDIR /app

COPY . .

CMD ['python3', '-m', 'flask', 'run', '--host=0.0.0.0']