# BASE STAGE ==============================================
FROM nvidia/cuda:12.6.3-base-ubuntu24.04 AS base

ARG APP_PATH="/arctic_rush"

ARG USERNAME=arctic_rush
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV SHELL=/bin/bash \
    PYTHONIOENCODING=utf8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user (base image ships an "ubuntu" user at uid/gid 1000)
RUN (getent passwd $USER_UID && userdel -r $(getent passwd $USER_UID | cut -d: -f1)) || true && \
    (getent group $USER_GID && groupdel $(getent group $USER_GID | cut -d: -f1)) || true && \
    groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

WORKDIR ${APP_PATH}


# BUILDER BASE ============================================
FROM base AS builder-base

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/.venv \
    UV_PYTHON_INSTALL_DIR=/opt/.uv-python \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=automatic

# uv manages its own Python 3.13 toolchain per pyproject.toml — install deps
# (cu126 torch index) into a cache-mounted layer before copying source.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --frozen --no-install-project

ENV PATH="/opt/.venv/bin:$PATH"


# TRAIN ====================================================
FROM builder-base AS train

COPY --chown=$USERNAME:$USERNAME src/ ./src/
COPY --chown=$USERNAME:$USERNAME assets/ ./assets/

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    SDL_VIDEODRIVER=dummy \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    RUN_ID=default \
    MODEL_DIR=/data/models \
    LOG_DIR=/data/logs

# No seed checkpoint. The image used to bake a leela.pth in as the "default"
# run's starting weights, but that checkpoint predates the network rewrite --
# different action space, observation shape, trunk and heads -- so loading it
# now fails outright, and the semantics it learned were wrong anyway.
# Every RUN_ID gets its own subdir under the mounted /data volume and starts
# from a freshly initialised network.
RUN mkdir -p /data/models /data/logs && \
    chown -R $USERNAME:$USERNAME /data

USER $USERNAME

CMD ["python", "-m", "src.core.train"]
