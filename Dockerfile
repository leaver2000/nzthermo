# syntax=docker/dockerfile:1

# .................................................................................................
FROM python:3.11.9 AS py311
USER root
WORKDIR /

ENV PATH="/opt/venv/bin:$PATH"
RUN python3 -m venv /opt/venv
COPY tests/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV NZTHERMO_BUILD_COVERAGE=1
RUN pip install --no-cache-dir --no-deps --upgrade --target src/ . \
  && pytest tests

# numpy 2.0.0 testing
RUN pip install --no-cache-dir Pint==0.24 numpy==2.0.0 --upgrade \
  && pytest tests

USER 1001

# .................................................................................................
FROM python:3.12.3 AS py312
USER root
WORKDIR /

ENV PATH="/opt/venv/bin:$PATH"
RUN python3 -m venv /opt/venv
COPY tests/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV NZTHERMO_BUILD_COVERAGE=1
RUN pip install --no-cache-dir --no-deps --upgrade --target src/ . \
  && pytest tests

# numpy 2.0.0 testing
RUN pip install --no-cache-dir Pint==0.24 numpy==2.0.0 --upgrade \
  && pytest tests

USER 1001
