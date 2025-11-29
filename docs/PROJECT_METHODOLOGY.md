# Project Methodology and Workflow

This document describes the development philosophy adopted in this Machine Learning project, aligned with best practices in software engineering and data science in mature corporate environments.

## The Philosophy: Laboratory vs. Factory

Our approach divides work into two distinct environments, each with clear purposes:

### 1. The Laboratory (Jupyter Notebooks)

* **Location:** `notebooks/` directory
* **Purpose:** Discovery, rapid experimentation, and hypothesis validation.
* **Characteristics:**
  * Focus on agility and visualization.
  * Allows controlled "messiness" (out-of-order cells, multiple prints, exploratory graphs).
  * Answers questions like: "Does variable X correlate with the target?", "What is the Dollar's impact on sales?".
* **Project Example:** Notebooks `2.0-lab-salesforce-funnel-analysis.ipynb` and `3.0-lab-market-correlation-analysis.ipynb`.

### 2. The Factory (Source Code)

* **Location:** `src/` directory
* **Purpose:** Production, automation, reproducibility, and scale.
* **Characteristics:**
  * Clean, modular, and object-oriented code.
  * Testable and versioned.
  * Ready to be integrated into CI/CD pipelines.
* **Project Example:** Scripts in `src/data/`, `src/features/`, and `src/pipeline.py`.

---

## The Development Cycle (The Loop)

Development is not linear, but rather an iterative cycle of continuous improvement:

1. **Exploration (Notebook):**

   * The data scientist tests a new idea (e.g., creating a moving average feature).
   * Validates whether the idea brings performance gains to the model.
2. **Refactoring (Script):**

   * If the idea was validated, the logic is extracted from the notebook.
   * The code is rewritten in a robust and modular way within `src/` (e.g., creating a new class in `src/features`).
3. **Integration (Pipeline):**

   * The main pipeline (`src/pipeline.py`) is updated to use the new official component.
   * The model is retrained automatically.
4. **Restart:**

   * Return to the notebook to explore the next hypothesis.

## Benefits of This Approach

* **Reproducibility:** Anyone (or any machine) can run the pipeline and obtain the same result.
* **Maintainability:** Bugs are fixed in a single place (`src/`), not scattered across dozens of notebooks.
* **Scalability:** Code in `src/` can be easily deployed to production (APIs, Batch Jobs), while notebooks are difficult to operationalize.

---

## Visual Diagrams

### Philosophy: Laboratory vs. Factory

```mermaid
graph LR
    subgraph WORKSPACE[PROJECT WORKSPACE]
        subgraph LAB[LABORATORY - notebooks]
            L1[Jupyter Notebooks]
            L2[Experimentation]
            L3[Visualization]
            L4[Quick Prototypes]
            L5[Hypothesis Testing]
        end

        subgraph FAC[FACTORY - src]
            F1[Python Modules]
            F2[Clean Code]
            F3[Unit Tests]
            F4[Documentation]
            F5[Type Hints]
        end
    end

    LAB --- FAC

    style LAB fill:#e1f5fe,stroke:#01579b
    style FAC fill:#e8f5e9,stroke:#1b5e20
```

### Data Flow Pipeline

```mermaid
graph LR
    subgraph RAW[RAW DATA]
        R1[Salesforce]
        R2[Macro]
        R3[B3/IBGE]
    end

    subgraph PROCESSED[PROCESSED DATA]
        P1[Cleaned]
        P2[Validated]
        P3[Typed]
    end

    subgraph FEATURES[FEATURES]
        F1[Engineered]
        F2[Selected]
        F3[Scaled]
    end

    subgraph MODEL[MODEL]
        M1[Fit]
        M2[Evaluate]
        M3[Export]
    end

    RAW -->|Load| PROCESSED
    PROCESSED -->|Transform| FEATURES
    FEATURES -->|Train| MODEL

    RAW -.->|data/raw/| DB1[(Storage)]
    PROCESSED -.->|data/processed/| DB2[(Storage)]
    FEATURES -.->|data/features/| DB3[(Storage)]
    MODEL -.->|models/| DB4[(Storage)]

    style RAW fill:#ffebee,stroke:#c62828
    style PROCESSED fill:#e3f2fd,stroke:#1565c0
    style FEATURES fill:#e8f5e9,stroke:#2e7d32
    style MODEL fill:#f3e5f5,stroke:#7b1fa2
```

### Benefits Summary

```mermaid
graph TB
    subgraph BENEFITS[KEY BENEFITS]
        subgraph REPRO[REPRODUCIBILITY]
            RE1[Same inputs = Same outputs]
            RE2[Version controlled experiments]
            RE3[Automated pipelines]
        end

        subgraph MAINT[MAINTAINABILITY]
            MA1[Single source of truth]
            MA2[Easy bug fixes]
            MA3[Clear ownership]
        end

        subgraph SCALE[SCALABILITY]
            SC1[Production-ready code]
            SC2[API deployment]
            SC3[Batch job ready]
        end
    end

    style REPRO fill:#e3f2fd,stroke:#1565c0
    style MAINT fill:#fff3e0,stroke:#ef6c00
    style SCALE fill:#e8f5e9,stroke:#2e7d32
```

<style>#mermaid-1764448566185{font-family:sans-serif;font-size:16px;fill:#333;}#mermaid-1764448566185 .error-icon{fill:#552222;}#mermaid-1764448566185 .error-text{fill:#552222;stroke:#552222;}#mermaid-1764448566185 .edge-thickness-normal{stroke-width:2px;}#mermaid-1764448566185 .edge-thickness-thick{stroke-width:3.5px;}#mermaid-1764448566185 .edge-pattern-solid{stroke-dasharray:0;}#mermaid-1764448566185 .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-1764448566185 .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-1764448566185 .marker{fill:#333333;}#mermaid-1764448566185 .marker.cross{stroke:#333333;}#mermaid-1764448566185 svg{font-family:sans-serif;font-size:16px;}#mermaid-1764448566185 .label{font-family:sans-serif;color:#333;}#mermaid-1764448566185 .label text{fill:#333;}#mermaid-1764448566185 .node rect,#mermaid-1764448566185 .node circle,#mermaid-1764448566185 .node ellipse,#mermaid-1764448566185 .node polygon,#mermaid-1764448566185 .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#mermaid-1764448566185 .node .label{text-align:center;}#mermaid-1764448566185 .node.clickable{cursor:pointer;}#mermaid-1764448566185 .arrowheadPath{fill:#333333;}#mermaid-1764448566185 .edgePath .path{stroke:#333333;stroke-width:1.5px;}#mermaid-1764448566185 .flowchart-link{stroke:#333333;fill:none;}#mermaid-1764448566185 .edgeLabel{background-color:#e8e8e8;text-align:center;}#mermaid-1764448566185 .edgeLabel rect{opacity:0.5;background-color:#e8e8e8;fill:#e8e8e8;}#mermaid-1764448566185 .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#mermaid-1764448566185 .cluster text{fill:#333;}#mermaid-1764448566185 div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:sans-serif;font-size:12px;background:hsl(80,100%,96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#mermaid-1764448566185:root{--mermaid-font-family:sans-serif;}#mermaid-1764448566185:root{--mermaid-alt-font-family:sans-serif;}#mermaid-1764448566185 flowchart{fill:apa;}</style>
