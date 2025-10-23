# AURORA Architecture Diagram

Below diagram is rendered via Mermaid (GitHub renders this automatically).  
It reflects the multi‑agent design, observability stack, human‑in‑the‑loop, and runtime assumptions described in the attached paper.

```mermaid
%% AURORA System Architecture (from AgenticRCA paper)
flowchart LR
  subgraph Observability["Observability Plane"]
    OTel[OpenTelemetry Collectors]
    Prom[Prometheus]
    ES[Elasticsearch]
    Jaeger[Jaeger Tracing]
    OTel --> Prom
    OTel --> ES
    OTel --> Jaeger
  end

  subgraph API["AURORA API"]
    APIGW[FastAPI / REST]
  end

  subgraph Agents["AURORA Multi‑Agent System"]
    SUP[Supervisor Agent (ReAct)]
    TEL[Telemetry Agent]
    ANO[Anomaly Detection Agent]
    CAU[Causal Inference Agent\n(Hierarchical RCD + Neural Granger)]
    LOC[Root Cause Localization Agent\n(PageRank + Uncertainty)]
    REM[Remediation Agent\n(K8s HPA/VPA, Rollback, Restart)]
    LRN[Learning Agent\n(Knowledge Base / RL)]
  end

  subgraph Data["State & Knowledge"]
    PG[(PostgreSQL)]
    R[(Redis)]
    VDB[(Vector DB / Qdrant)]
    KB[(Incident KB)]
  end

  subgraph Runtime["Runtime Platform"]
    K8s[(Kubernetes Cluster)]
    HPA[HPA]
    VPA[VPA]
    K8s --- HPA
    K8s --- VPA
  end

  subgraph Human["Human‑in‑the‑Loop"]
    SRE[SRE / Operator]
    Graf[Grafana & Dashboards]
  end

  APIGW --> SUP
  SUP --> TEL
  SUP --> ANO
  SUP --> CAU
  SUP --> LOC
  SUP --> REM
  SUP --> LRN

  TEL --> Prom
  TEL --> ES
  TEL --> Jaeger

  ANO --> CAU
  CAU <--> LOC

  LOC --> APIGW
  REM --> K8s

  SUP <--> PG
  SUP <--> R
  LRN <--> KB
  LRN <--> VDB

  Prom --> Graf
  ES --> Graf
  Jaeger --> Graf
  SRE <--> Graf
  SRE <--> APIGW

  classDef plane fill:#f8fafc,stroke:#94a3b8,stroke-width:1px;
  class Observability,Agents,Data,Runtime,Human,API plane;
```

> Source: Derived from the attached AURORA/AgenticRCA research paper (see repository root for citation).
