# syntax=docker/dockerfile:1
#
# Two images from one file, because only one of the two roles needs a JVM.
#
#   --target cli      Python only. Generation, inspection, PR delivery.
#   --target worker   + JRE 17 and PySpark. Executes Spark, so it needs one.
#
# A JRE and PySpark are several hundred megabytes the generation path never
# touches, which is the whole reason for the split. Measured on a kind node:
# 135MB for `cli` against 667MB for `worker` -- so the pod that does not run
# Spark carries a fifth of the image, and none of the Java.
#
# Building both from a shared `base` stage keeps the dependency set identical
# between them — a worker whose pydantic differs from the CLI's would fail in
# ways that are miserable to diagnose.
#
# Notes on the choices here, since several are load-bearing:
#
# * **Non-root.** The whole point of the sandbox (`sandbox/runner.py`) is that
#   generated code is untrusted. Running the process that spawns it as uid 0
#   would hand back most of what the sandbox is protecting. `etlm` owns nothing
#   it does not need to write.
# * **JRE, not JDK.** PySpark launches a JVM; it does not compile Java. The JDK
#   is roughly twice the size and adds a compiler to an image that executes
#   untrusted input.
# * **No build tooling in the final layers.** Wheels are built in `builder` and
#   only the resulting virtualenv is copied forward, so `gcc` never ships.
# * **`PYTHONDONTWRITEBYTECODE`** keeps a read-only root filesystem viable in
#   Kubernetes; the writable paths are declared as volumes.

# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# build-essential is needed to compile any sdist-only dependency; it stays in
# this stage and is never copied into a runtime image.
RUN apt-get update \
 && apt-get install --no-install-recommends -y build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependency metadata first: this layer is cached until the *dependencies*
# change, so an edit to a source file does not re-resolve the whole tree.
COPY pyproject.toml README.md ./
COPY src/ ./src/

ARG INSTALL_SPARK=false
RUN if [ "$INSTALL_SPARK" = "true" ]; then \
        pip install --no-cache-dir ".[spark]" ; \
    else \
        pip install --no-cache-dir "." ; \
    fi

# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    ETLM_WORKSPACE_DIR=/var/lib/etl-migrator/workspace

# A fixed uid/gid so a Kubernetes securityContext can name it, and so files
# written to a mounted volume have a predictable owner.
RUN groupadd --gid 10001 etlm \
 && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin etlm \
 && mkdir -p /var/lib/etl-migrator/workspace \
 && chown -R etlm:etlm /var/lib/etl-migrator

COPY --from=builder --chown=root:root /opt/venv /opt/venv
COPY --chown=root:root examples/ /app/examples/
COPY --chown=root:root fixtures/ /app/fixtures/

WORKDIR /app
USER etlm

# The workspace is the only path the process writes to, so it is the only one
# that needs to be writable when the root filesystem is mounted read-only.
VOLUME ["/var/lib/etl-migrator/workspace"]

# ---------------------------------------------------------------------------
FROM base AS cli

LABEL org.opencontainers.image.title="autonomous-etl-migration-agent (cli)" \
      org.opencontainers.image.description="Migrates legacy ETL pipelines to validated PySpark. No JVM: generation and delivery only." \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/1mad-elmakaoui/autonomous-etl"

# Proves the package imports and the CLI is wired, without needing a JVM or a
# network. A container that cannot answer this is not serving anything.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["etl-migrator", "--help"]

ENTRYPOINT ["etl-migrator"]
CMD ["--help"]

# ---------------------------------------------------------------------------
FROM base AS worker

USER root
# Spark 4 requires Java 17 or later. Bookworm's `main` carries 17 and not 21 --
# 21 first ships with trixie -- so this is 17, and the version here has to track
# whatever the pinned base image actually provides rather than whatever the
# development machine happens to run. Bumping the base without bumping this is
# the exact mistake `test_the_declared_java_home_matches_the_installed_jdk`
# exists to catch.
RUN apt-get update \
 && apt-get install --no-install-recommends -y openjdk-17-jre-headless procps \
 && rm -rf /var/lib/apt/lists/*
USER etlm

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    SPARK_LOCAL_DIRS=/var/lib/etl-migrator/workspace/spark-tmp

LABEL org.opencontainers.image.title="autonomous-etl-migration-agent (worker)" \
      org.opencontainers.image.description="Temporal worker: executes both pipelines in a sandbox, diffs, benchmarks and delivers." \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/1mad-elmakaoui/autonomous-etl"

# A worker is healthy when it can reach Temporal, which `--help` cannot tell
# us. Checking the JVM instead catches the failure this image is uniquely
# exposed to: a broken or missing Java installation, which otherwise surfaces
# only when the first Spark activity runs, minutes into a migration.
HEALTHCHECK --interval=30s --timeout=15s --start-period=20s --retries=3 \
    CMD ["java", "-version"]

ENTRYPOINT ["etl-migrator"]
CMD ["worker"]
