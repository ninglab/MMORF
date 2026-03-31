FROM python:3.11.1-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository
# RUN git clone https://github.com/ninglab/MMORF.git .
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -e .
RUN cd mmorf && pip install --no-cache-dir -e rdchiral && cd ..

# Create data and output directories
RUN mkdir -p saved_model prepare_data

# Note: You must manually download and place the required files:
# - saved_model/policy_model.ckpt
# - saved_model/template_rules.dat
# - saved_model/value_pc.pt
# Copy required model and data files
# COPY saved_model/policy_model.ckpt saved_model/
# COPY saved_model/template_rules.dat saved_model/
# COPY saved_model/value_pc.pt saved_model/
# COPY prepare_data/origin_dict.csv prepare_data/

RUN cd mmorf/prepare_data && gunzip purch_cost.json.gz

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/bin/bash"]