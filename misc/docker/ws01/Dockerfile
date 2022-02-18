# Instructions copied from - https://hub.docker.com/_/python/
FROM python:3

EXPOSE 5000

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "./app.py"]
