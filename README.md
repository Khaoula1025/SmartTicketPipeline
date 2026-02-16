# 🎫 NLP Support Ticket Classification Pipeline with MLOps

## 📋 Project Overview

An end-to-end MLOps pipeline for automated classification of IT support tickets using Natural Language Processing. This project implements a production-ready batch processing system that analyzes customer emails, generates semantic embeddings, and predicts ticket categories while maintaining robust monitoring and infrastructure management.

### Key Features

- 🤖 **NLP Text Processing**: Advanced text preprocessing and tokenization
- 🧠 **Semantic Embeddings**: Hugging Face transformer models for vector representations
- 📊 **Classification Model**: Supervised learning for ticket type prediction
- 🗄️ **Vector Database**: ChromaDB for efficient embedding storage and retrieval
- 📈 **ML Monitoring**: Evidently AI for data drift and model performance tracking
- 🐳 **Containerization**: Full Docker and Kubernetes orchestration
- 📡 **Infrastructure Monitoring**: Prometheus & Grafana for system health
- 🔄 **CI/CD**: Automated pipeline with GitHub Actions

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Email Tickets  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ NLP Preprocessing│
│  - Tokenization │
│  - Normalization│
│  - Stopwords    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Hugging Face    │
│ Embeddings      │
└────────┬────────┘
         │
         ├──────────────┐
         ▼              ▼
┌─────────────────┐  ┌──────────────┐
│   ChromaDB      │  │Classification│
│ Vector Storage  │  │    Model     │
└─────────────────┘  └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Predictions │
                     └──────┬───────┘
                            │
         ┌──────────────────┴──────────────────┐
         ▼                                     ▼
┌─────────────────┐                   ┌──────────────┐
│  Evidently AI   │                   │  Prometheus  │
│ (ML Monitoring) │                   │   Grafana    │
└─────────────────┘                   └──────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Docker & Docker Compose
- Kubernetes (minikube)
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Khaoula1025/SmartTicketPipeline.git
cd nlp-ticket-classification
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

---

## 📂 Project Structure

```
.
├── data/
│   ├── raw/                    # Original email datasets
│   └── processed/              # Cleaned and preprocessed data
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── preprocessing/
│   │   └── nlp_cleaner.py     # Text cleaning and tokenization
│   ├── embeddings/
│   │   └── generator.py       # Hugging Face embeddings
│   ├── models/
│       └── classifier.py      # Classification model training
|
├── kubernetes/
│   ├── job.yaml               # Kubernetes Job definition
│   └── cronjob.yaml           # Scheduled pipeline execution
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       └── dashboards/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci-cd.yaml         # GitHub Actions pipeline
├── reports/
│   └── evidently/             # Generated drift reports
├── models/
│   └── saved_models/          # Trained model artifacts
├── requirements.txt
└── README.md
```

---

## 🔄 Pipeline Steps

### Step 1: Exploratory Analysis & NLP Preprocessing

**Objective**: Prepare textual data for machine learning

- Analyze ticket type distribution
- Measure email length statistics
- Merge text fields (subject + body)
- Apply NLP cleaning:
  - Lowercase normalization
  - Punctuation removal
  - Tokenization
  - Stopword removal

```python
python src/preprocessing/nlp_cleaner.py --input data/raw/tickets.csv --output data/processed/
```

### Step 2: Embedding Generation

**Objective**: Transform text into semantic vectors using Hugging Face

- Load pre-trained transformer model
- Encode cleaned text into embeddings
- Normalize vectors
- Index in ChromaDB

```python
python src/embeddings/generator.py --model sentence-transformers/all-MiniLM-L6-v2
```

### Step 3: Model Training

**Objective**: Train supervised classifier for ticket type prediction

- Split train/test datasets
- Train with scikit-learn
- Evaluate performance (Precision, Recall, F1-Score)

```python
python src/models/classifier.py --train
```

### Step 5: ML Monitoring with Evidently AI

**Objective**: Detect data and prediction drift

```python
from evidently import Report
from evidently.presets import ClassificationPreset, DataDriftPreset

# Generate drift report
report = Report(metrics=[
    DataDriftPreset(),
    ClassificationPreset()
])

report.run(reference_data=baseline, current_data=new_data)
report.save_html("reports/evidently/drift_report.html")
```

### Step 6: Containerization & Orchestration

**Build Docker image**
```bash
docker build -t nlp-pipeline:latest -f docker/Dockerfile .
```

**Deploy to Kubernetes**
```bash
# Start minikube
minikube start

# Apply Kubernetes manifests
kubectl apply -f kubernetes/job.yaml

# For scheduled execution
kubectl apply -f kubernetes/cronjob.yaml
```

### Step 7: Infrastructure Monitoring

**Start monitoring stack**
```bash
cd monitoring
docker-compose up -d
```

Access services:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (default: admin/admin)
- **cAdvisor**: http://localhost:8080

---

## 📊 Monitoring & Observability

### ML Monitoring (Evidently AI)

- **Data Drift**: Detect distribution changes in input features
- **Prediction Drift**: Track model output stability
- **Performance Metrics**: Monitor accuracy, precision, recall over time

Reports are generated in `reports/evidently/` as interactive HTML files.

### Infrastructure Monitoring (Prometheus + Grafana)

**Metrics Collected:**
- CPU/RAM/Disk usage (Node Exporter)
- Container resource consumption (cAdvisor)
- Pipeline execution times
- Model inference latency

**Sample Prometheus Configuration:**
```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
  
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

---

## 🔧 CI/CD Pipeline

GitHub Actions workflow automates:

1. **Linting**: Code quality checks
2. **Testing**: Unit and integration tests
3. **Docker Build**: Container image creation
4. **Registry Push**: Image deployment
5. **Kubernetes Deployment**: Automated rollout

**Triggers:**
- Push to `main` or `develop` branches
- Pull request creation

---

## 📈 Performance Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| F1-Score | > 0.85 | ✅ |
| Data Preprocessing Quality | Complete pipeline | ✅ |
| ChromaDB Integration | Functional semantic search | ✅ |
| Evidently Reports | Drift detection operational | ✅ |
| Prometheus/Grafana | All metrics collected | ✅ |
| Kubernetes Deployment | No errors | ✅ |

---

## 🛠️ Technologies Used

### Core ML Stack
- **Python 3.8+**
- **Hugging Face Transformers**: Embedding generation
- **scikit-learn**: Classification models
- **ChromaDB**: Vector database
- **Evidently AI**: ML monitoring

### Infrastructure
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **cAdvisor**: Container monitoring
- **Node Exporter**: System metrics

### DevOps
- **GitHub Actions**: CI/CD automation
- **minikube**: Local Kubernetes cluster

---

## 📝 Deliverables

- ✅ NLP preprocessing scripts
- ✅ ChromaDB embeddings storage
- ✅ Trained classification model
- ✅ Evidently AI drift reports
- ✅ Docker images
- ✅ Kubernetes manifests
- ✅ Technical documentation

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is part of the **RNCP AI Developer Certification [2023]** program.

---

## 👤 Author

**Yassine Ennaya**  
Created: 07/02/26

---

## 📞 Support

For questions or issues, please open an issue in the GitHub repository or contact the project maintainer.

---

## 🎯 Project Timeline

- **Duration**: 1 week
- **Start**: 09/02/2026
- **Deadline**: 14/02/2026 (midnight)
- **Format**: Individual work

---

## ⚡ Quick Start Commands

```bash
# Complete pipeline execution
make run-pipeline

# Run preprocessing only
make preprocess

# Train model
make train

# Generate monitoring reports
make monitoring

# Deploy to Kubernetes
make deploy

# Start monitoring stack
make monitoring-up
```

---

**Built with ❤️ for automated IT support ticket classification**