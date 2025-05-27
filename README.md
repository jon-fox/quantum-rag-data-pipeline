# quantum-rag-data-pipeline

This project is the data pipeline responsible for fetching, processing, and storing data for the Quantum Semantic Reranking in RAG Pipelines project.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Resources

These resources pertain to the main Quantum RAG project for which this pipeline supplies data:

| Resource | Link |
|----------|------|
| Qiskit (Quantum SDK) | https://github.com/Qiskit/qiskit |
| Project Discussion | https://chatgpt.com/c/68254a64-f578-8001-b942-33e437225165 |
| Original Proposal | https://docs.google.com/document/d/19WuIULxvqFG6xaQ2Sa7sYMlx8o4hZBwX4khceGAqRag/edit?tab=t.0 |
| Atomic Agents | https://github.com/BrainBlend-AI/atomic-agents |
| uv (package) manager | https://github.com/astral-sh/uv |

## Project Objective

This `quantum-rag-data-pipeline` project serves as the dedicated data ingestion and storage component for a graduate independent study on Quantum Semantic Reranking in RAG Pipelines. Its primary responsibilities are:

- Fetching energy forecast data from the ERCOT (Electric Reliability Council of Texas) API.
- Potentially integrating other relevant data sources (e.g., weather data via `src/data/weather_api/weather.py`).
- Processing and preparing this data for semantic search and analysis.
- Storing the processed data, including embeddings, into a pgvector database (utilizing `src/storage/pgvector_storage.py`).

The data curated by this pipeline is intended for use by the main quantum-enhanced RAG system to perform energy forecast analysis and market trend predictions.

## ERCOT API Integration

The project includes integration with the ERCOT API for accessing real-time and historical energy data:

- Authentication with automatic token refresh every 55 minutes (tokens expire after 60 minutes) (see `src/data/ercot_api/auth.py`).
- Access to real-time pricing data, historical load data, and forecasts (see `src/data/ercot_api/client.py` and `src/data/ercot_api/queries.py`).
- Environment variable management for secure credential storage (see `src/config/env_manager.py`).

### Setup ERCOT API Credentials

1. Create a `.env` file in the project root (or ensure your environment variables are set).
2. Add your ERCOT API credentials to the `.env` file or environment:
   ```
   ERCOT_API_USERNAME=your-username
   ERCOT_API_PASSWORD=your-password
   ```
3. The application, via `src/config/env_manager.py`, will load these credentials.

### Example ERCOT API Usage

The `src/data/ercot_api/` modules can be used to fetch data. An example of how to use the client and queries might look like this (conceptual, adapt as needed):

```python
# (Assuming ERCOTClient and ERCOTQueries are properly imported and initialized)
# from src.data.ercot_api.client import ERCOTClient
# from src.data.ercot_api.queries import ERCOTQueries
# from src.config.env_manager import EnvManager

# EnvManager.load_env() # To load credentials from .env
# client = ERCOTClient() # Client handles authentication

# Initialize ERCOT queries helper
# queries = ERCOTQueries(client)

# Get 2-Day Aggregated Generation Summary
# gen_summary = queries.get_aggregated_generation_summary(
#     delivery_date_from="2025-05-17",
#     delivery_date_to="2025-05-18"
# )
# print(gen_summary)

# Get 2-Day Aggregated Load Summary for Houston region
# load_houston = queries.get_aggregated_load_summary(
#     delivery_date_from="2025-05-17",
#     delivery_date_to="2025-05-18",
#     region="Houston"
# )
# print(load_houston)
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/quantum-rag-data-pipeline.git # Replace with your actual repo URL
cd quantum-rag-data-pipeline

# Create and activate a virtual environment (recommended)
# python -m venv venv
# source venv/bin/activate # On Windows: venv\Scripts\activate

# Install dependencies
# Create a requirements.txt file with necessary packages like:
# psycopg2-binary (for pgvector)
# boto3 (if using DynamoDB, e.g. for src.storage.dual_storage)
# requests
# python-dotenv
# pandas (often useful for data manipulation)
# And then run:
# pip install -r requirements.txt

# Setup ERCOT API Credentials (as described in the ERCOT API Integration section)
# Ensure your PostgreSQL server with pgvector extension is running and configured.

# Run data ingestion and processing scripts
# Scripts are located in src/scripts/
# Example: python src/scripts/create_weather_table.py
# (You will need to develop further scripts for fetching ERCOT data and storing it in pgvector)
```

This data pipeline is a foundational component for exploring the application of quantum computation within NLP pipelines, specifically for energy system forecasting and operational analysis.
