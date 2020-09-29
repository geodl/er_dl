FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

COPY ./requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt

COPY ./project /app/project
ENV PYTHONPATH /app/project/:$PYTHONPATH
RUN find /app/project

WORKDIR /app
ADD ./project/__init__.py ./project/__init__.py
COPY setup.py .
RUN pip install -e .
WORKDIR /app/project/app/
