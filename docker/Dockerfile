FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# ULCC symlink rule (see AGENTS.md): ensure import-name vs folder-name symlinks
RUN ln -s /app/ulcc-core /app/ulcc_core && ln -s /app/ulcc-ddg /app/ulcc_ddg && ln -s /app/pggs-toy /app/ulcc_pggs
CMD ["pytest", "-q"]
