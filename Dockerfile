FROM python:3.10.1-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

RUN pip install --no-cache-dir -r requirements.txt

# RUN apt-get update && apt-get install -y <package_name>
