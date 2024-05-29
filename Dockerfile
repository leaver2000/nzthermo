# syntax=docker/dockerfile:1

# .................................................................................................
FROM python:3.10.14
USER root
WORKDIR /

ENV PATH="/opt/venv/bin:$PATH"
RUN python3 -m venv /opt/venv
COPY tests/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV NZTHERMO_BUILD_COVERAGE 1
RUN pip install --no-cache-dir --no-deps --upgrade --target src/ .

USER 1001

# .................................................................................................
FROM python:3.11.9
USER root
WORKDIR /

ENV PATH="/opt/venv/bin:$PATH"
RUN python3 -m venv /opt/venv
COPY tests/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV NZTHERMO_BUILD_COVERAGE 1
RUN pip install --no-cache-dir --no-deps --upgrade --target src/ .

USER 1001

# .................................................................................................
FROM python:3.12.3
USER root
WORKDIR /

ENV PATH="/opt/venv/bin:$PATH"
RUN python3 -m venv /opt/venv
COPY tests/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV NZTHERMO_BUILD_COVERAGE 1
RUN pip install --no-cache-dir --no-deps --upgrade --target src/ .

USER 1001
