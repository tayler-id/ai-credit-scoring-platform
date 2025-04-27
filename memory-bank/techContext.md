# Tech Context: AI-Based Credit Scoring Platform

## Technologies Used (Initial Setup Choices)
*   **Programming Languages:** Python 3.10 (as per Dockerfile).
*   **Cloud Platform:** Tentatively AWS (S3, RDS, SageMaker considered). Decision pending confirmation if needed.
*   **Databases:** Initial plan includes PostgreSQL (structured data) and S3/Data Lake (raw data). `psycopg2-binary` added to requirements.
*   **Core Libraries:**
    *   `fastapi`: API framework.
    *   `uvicorn`: ASGI server.
    *   `pydantic`: Data validation and settings management.
    *   `pandas`, `numpy`, `scikit-learn`: Core data science stack.
    *   `python-dotenv`: Environment variable loading.
*   **Data Processing:** TBD (Initial setup doesn't include specific processing tools yet).
*   **Pipeline Orchestration:** TBD (Airflow/Prefect considered).
*   **API Framework:** FastAPI (confirmed).
*   **Containerization:** Docker (Dockerfile created).
*   **CI/CD:** GitHub Actions (CI workflow implemented).
*   **Orchestration:** TBD (Initial setup uses basic Docker).
*   **MLOps Platforms:** TBD.
*   **Monitoring:** TBD.
*   **Infrastructure as Code (IaC):** TBD (Terraform considered, `infrastructure/` directory created).

## Development Setup
*   **Version Control:** Git (Assumed, repo hosting TBD). `.gitignore` created.
*   **CI/CD:** GitHub Actions (CI workflow at `.github/workflows/ci.yml`).
*   **Local Environment:** Docker (via Dockerfile). Virtual environments recommended. `requirements.txt` created for dependency management.
*   **IDE:** VS Code recommended.
*   **Testing:** `pytest`, `pytest-cov`, `requests` added to requirements. Basic API tests created in `tests/api/`.
*   **Linting/Formatting:** `flake8`, `black` added to requirements. (Integration into workflow TBD).
*   **Configuration:** Pydantic `BaseSettings` (`config/settings.py`) loading from `.env` file (`.env.example` provided).

## Technical Constraints
*   Data privacy and security regulations (e.g., GDPR-like laws in target markets) are paramount.
*   Scalability requirements for data volume and API requests.
*   Latency requirements for scoring API.
*   Model explainability and fairness requirements.
*   Potential limitations in data availability and quality in target markets.
*   Cost-effectiveness of cloud services.

## Dependencies
*   External data source APIs.
*   Cloud provider services.
*   Open-source libraries (need careful management).

## Tool Usage Patterns
*   Use `sequentialthinking` MCP for planning.
*   Use `context7` MCP for library documentation lookups.
*   Follow Memory Bank protocol for documentation.
*   Use IaC for infrastructure management.
*   Automate testing and deployment via CI/CD.

*(Reflects choices made during initial project scaffolding. Will be updated as more components are built and decisions finalized.)*
