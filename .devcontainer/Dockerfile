FROM mcr.microsoft.com/devcontainers/python:2-3.13-trixie

# Verify Python version
RUN python --version

RUN pip install --no-cache-dir pipx \
    && pipx reinstall-all \
    && pipx install uv

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

CMD ["bash"]