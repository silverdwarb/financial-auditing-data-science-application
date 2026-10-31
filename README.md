# financial-auditing-data-science-application

## Project Overview

The **inancial-auditing-data-science-application** is a Python-based forensic accounting and data analytics project designed to identify potentially fraudulent or unusual journal entries through statistical analysis, anomaly detection, and network-based risk propagation.

The project aims to explore how traditional audit procedures can be enhanced through data analytics and machine learning. Rather than relying on a single fraud detection method, the system will use multiple independent analytical techniques to identify risk indicators, assign risk scores, and prioritize journal entries for further investigation.

The long-term goal is to move beyond evaluating individual transactions and explore how suspicious activity may propagate through an organization's financial relationships.

**Core concept:** Start with traditional forensic accounting analytics, build a risk propagation model that identifies potentially connected suspicious activity, and eventually develop a neural network that learns patterns associated with fraudulent journal entries.

> **Important:** A high risk score is not proof of fraud. The system is designed to support audit investigation and risk assessment, not replace professional judgment or established audit procedures.

---

## Project Objectives

- Acquire and prepare datasets containing fraudulent and nonfraudulent journal entries.
- Develop a modular forensic analytics engine with independently executable tests.
- Assign individual risk scores based on different analytical techniques.
- Model how risk signals can propagate between entities through financial interactions.
- Explore whether a neural network can identify complex patterns associated with fraudulent journal entries.
- Produce interpretable results that help an auditor understand why an entry was flagged.

---

## Project Roadmap

### Beta — Traditional Forensic Analytics Engine

**Objective:** Establish the data pipeline and build a terminal-based application that allows users to select and execute individual forensic analytics scripts.

The initial version focuses on simpler, interpretable techniques before introducing network propagation or machine learning.

#### 1. Dataset acquisition and preparation

Acquire datasets containing both fraudulent and nonfraudulent journal entries. Investigate publicly available research datasets and synthetic accounting datasets that include fraud labels or known anomalies.

The data preparation pipeline should:

- Load journal-entry data from CSV files.
- Validate required columns and data types.
- Handle missing values, inconsistent formats, and duplicate records.
- Standardize dates, amounts, debit/credit indicators, and user identifiers.
- Preserve fraud labels separately from the analytical features.
- Document the source, limitations, and meaning of each dataset.

Potential journal-entry fields include:

| Field | Purpose |
|---|---|
| `posting_id` | Identify a journal entry or posting record |
| `document_date` | Date associated with the source document |
| `posting_date` | Accounting date of the entry |
| `entered_date` | Actual date and time the entry was recorded |
| `gl_account` | General ledger account identifier |
| `amount` | Monetary value of the posting |
| `debit_credit` | Debit or credit indicator |
| `user` | Person or system that entered the posting |
| `ref_id` | Reference linking related records |
| `text` | Journal-entry description |
| `fraud_label` | Known fraud label, when available |

The final schema will depend on the selected dataset. Some datasets represent a journal entry as multiple debit and credit posting lines, so the system must preserve the relationships between lines belonging to the same entry.

#### 2. Modular forensic analytics scripts

Each analytical method should be implemented as an independent Python module with its own output and risk score.

Initial techniques to investigate and implement include:

- Benford's Law analysis
- Round-number detection
- Near-threshold transaction detection
- Duplicate and near-duplicate journal entries
- Period-end and after-hours posting analysis
- Weekend and holiday posting analysis
- Unusual user-to-account relationships
- Unusual debit/credit account combinations
- Relative Size Factor (RSF) analysis
- Statistical outlier detection using Z-scores or the interquartile range
- Backdating and reversal analysis

Each module should return a standardized result containing the relevant journal-entry identifier, test name, score, explanation, and any supporting evidence.

Individual tests should remain independently executable so that their results can be compared without requiring the entire engine to run.

#### 3. Terminal-based user interface

Build a command-line interface that allows users to:

1. Load or select a dataset.
2. View available forensic analytics tests.
3. Select and run individual tests.
4. View the results and associated risk scores.
5. Export flagged entries and findings to CSV.

**Beta deliverable:** A working terminal-based forensic analytics tool with multiple independent tests and documented outputs.

---

### Version 1.0 — Organizational Fraud Risk Propagation

**Objective:** Develop a graph-based risk propagation system that models how suspicious activity may spread through an organization's financial relationships.

This phase introduces the project's central experimental concept: a *poison apple* or *infection model*.

Rather than treating every journal entry independently, the system will construct a financial relationship network from journal-entry data and investigate whether risk signals from one entity can help identify other potentially suspicious activity.

#### 1. Construct a financial relationship network

Using journal-entry records, derive relationships between available entities, such as:

- Employees or journal-entry preparers
- General ledger accounts
- Journal entries and posting groups
- Shared transaction references
- Other identifiable participants, where the dataset supports them

For example, two employees may be connected through a journal entry, shared posting reference, or recurring pattern of financial activity.

The network structure must reflect actual relationships present in the data. A connection should not be invented simply because two records appear near each other in a dataset.

#### 2. Establish baseline risk scores

Initialize each entity's risk score using evidence derived from the traditional forensic analytics engine.

For example:

- Unusual journal-entry amounts
- Repeated suspicious entries
- Unusual account combinations
- Abnormal posting times
- Existing fraud indicators

The initial score represents the entity's own observed behavior before incorporating propagated risk.

#### 3. Implement the infection model

Entities exceeding a configurable risk threshold become eligible to propagate risk signals through their financial relationships.

Each transaction or relationship acts as a potential transmission pathway. Connected entities receive additional risk according to configurable parameters, including:

- **Source risk:** The current risk level of the originating entity.
- **Transmission threshold:** The minimum source risk required to activate propagation.
- **Relationship strength:** How strongly the transaction connects the entities.
- **Transmission probability:** The proportion of risk transferred through a relationship.
- **Decay factor:** How risk decreases across multiple connections or over time.
- **Iteration limit:** How many propagation rounds the algorithm performs.
- **Risk ceiling:** A maximum score to prevent uncontrolled escalation.

A transaction may also receive an elevated investigation score when it connects to a high-risk entity, allowing the system to rank both people and financial records.

#### 4. Experiment with randomized parameters

Create a configurable simulation environment in which propagation parameters can be adjusted or randomly sampled within predefined ranges.

Experiment with variables such as:

- Transmission probability
- Activation threshold
- Risk decay
- Relationship weights
- Number of propagation iterations
- Initial risk distributions

Compare the resulting rankings under different configurations to examine how assumptions influence the detection of potentially connected suspicious activity.

Randomized parameters are intended to test the behavior and sensitivity of the model, not to manufacture evidence of fraud.

#### 5. Evaluate the model

Evaluate whether propagation helps prioritize suspicious journal entries beyond the original standalone tests.

Potential measurements include:

- Precision and recall against known fraud labels
- Precision among the highest-ranked entries
- False-positive rate
- Changes in risk rankings under different parameters
- Additional suspicious entries identified through network connections

The model should also be tested against legitimate high-volume business relationships to determine whether propagation creates excessive false positives.

**Version 1.0 deliverable:** A configurable graph-based infection model that combines baseline forensic risk scores with relationship-based risk propagation and produces ranked entities and journal entries for investigation.

---

### Version 2.0 — Neural Network Fraud Classification

**Objective:** Develop a small neural network that learns patterns associated with fraudulent journal entries using labeled historical data.

Unlike the infection model, which uses explicitly designed propagation rules, the neural network will learn statistical relationships between input features and known fraud labels during training.

The initial model should focus on individual journal-entry classification. A graph neural network could be explored later if the data and project scope justify it.

#### 1. Define the prediction target

The model's target variable will be a binary fraud label:

- `1` — Journal entry confirmed or labeled as fraudulent.
- `0` — Journal entry labeled as nonfraudulent.

Unlabeled entries should not automatically be treated as legitimate. Their handling will depend on the dataset and the evaluation design.

#### 2. Select input features

The network needs structured, measurable characteristics that describe each journal entry.

| Feature group | Example inputs | What the model can learn |
|---|---|---|
| Monetary | Amount, absolute amount, log-transformed amount | Unusual transaction sizes and amount patterns |
| Date and time | Posting hour, weekend indicator, month-end indicator | Unusual timing patterns |
| Entry metadata | Manual/automated source, document type, description length | Differences between entry types |
| Account behavior | Account identifier, account frequency, unusual account combinations | Abnormal account usage |
| User behavior | User's historical entry frequency, typical posting hours, normal account usage | Deviations from a user's typical activity |
| Statistical indicators | Benford deviation, round-number indicator, amount outlier score | Patterns identified by traditional forensic tests |
| Relationship features | Counterparty risk, connection count, propagated risk score | Whether an entry is associated with a suspicious financial network |

The actual features will depend on the available dataset. Some fields listed above may not be present, and some behavioral features will require historical records to calculate.

**Why these inputs matter:** Fraudulent entries are not necessarily distinguishable by their amounts alone. A relatively ordinary amount may be suspicious because of the user who posted it, the account combination, the timing, or its relationship to other activity. Combining multiple features gives the model more information than using a single rule.

#### 3. Build a small neural network

Begin with a simple feedforward neural network that accepts a fixed-length vector of numerical and encoded categorical features.

A possible architecture is:

- Input layer: engineered journal-entry features
- Hidden layer 1: 32 neurons with ReLU activation
- Hidden layer 2: 16 neurons with ReLU activation
- Output layer: 1 neuron with sigmoid activation

The output is a model-estimated fraud probability between 0 and 1.

A configurable classification threshold can convert that probability into a predicted class. The threshold should be selected based on validation results and the relative cost of missed fraud versus false positives.

Before using a neural network, establish a simpler baseline such as logistic regression or a decision tree. This will help determine whether the additional complexity improves performance.

#### 4. Train and evaluate the model

The model will be trained on labeled data and evaluated on separate data that it has not seen during training.

Key considerations include:

- Splitting data by time or related entity when appropriate to reduce data leakage.
- Handling class imbalance because fraudulent entries may be much less common than legitimate entries.
- Avoiding features that directly reveal the fraud label or were created after the fraud investigation.
- Comparing precision, recall, F1 score, precision-recall AUC, and false-positive rates.
- Evaluating performance at the top of the risk ranking, where auditors would realistically investigate entries.
- Checking whether the model generalizes beyond synthetic or narrowly defined datasets.

Accuracy alone will not be sufficient to establish model quality.

#### 5. Integrate model predictions

Once evaluated, the neural network can be integrated into the existing application as another independent analytical module.

Its output can be compared with the traditional forensic tests and the infection model to determine:

- Which entries each approach flags
- Whether the approaches identify different types of suspicious activity
- Whether network-derived features improve classification
- Whether the model improves the ranking of known fraudulent entries

**Version 2.0 deliverable:** A small, trained neural network that estimates fraud probabilities from journal-entry features, with documented evaluation results and integration into the existing forensic analytics application.

---

## Proposed Technical Stack

| Component | Technology | Purpose |
|---|---|---|
| Programming language | Python | Core application |
| Data processing | pandas, NumPy | Load, clean, and transform journal-entry data |
| Statistical analysis | SciPy, scikit-learn | Statistical tests, outlier detection, baseline models, evaluation |
| Graph modeling | NetworkX | Represent financial relationships and implement risk propagation |
| Neural network | PyTorch or TensorFlow/Keras | Train the Version 2.0 classifier |
| Storage | CSV initially; SQLite if needed | Store datasets and analytical results |
| User interface | Python CLI | Select and execute individual tests |
| Reporting | CSV, Excel-compatible outputs | Export risk rankings and supporting evidence |

The technology stack may evolve as the project develops. The early versions should prioritize interpretability and maintainability over unnecessary infrastructure.

---

## Expected Output

The system should ultimately produce a standardized report containing fields such as:

| Field | Description |
|---|---|
| `posting_id` | Identifier for the flagged journal entry |
| `baseline_risk` | Risk derived from standalone analytical tests |
| `propagated_risk` | Additional risk associated with network relationships |
| `model_probability` | Estimated fraud probability from the trained model, when available |
| `final_risk_score` | Configured score used to prioritize investigation |
| `triggered_tests` | Analytical tests that flagged the entry |
| `risk_explanation` | Human-readable explanation of the findings |
| `review_status` | Optional auditor review outcome |

The output should preserve the individual test results so that a reviewer can understand why an entry was flagged instead of receiving only an unexplained aggregate score.

The final risk score should be designed carefully to avoid double-counting highly correlated indicators, treating propagated suspicion as confirmed fraud, or presenting an uncalibrated score as a true probability.

---

## Project Scope and Limitations

- **Risk is not proof:** Unusual activity and network proximity are indicators for investigation, not evidence of wrongdoing on their own.
- **Dataset quality matters:** Model performance depends on the reliability of the underlying journal-entry records and fraud labels.
- **Synthetic data has limitations:** Simulated results may not generalize to real organizational accounting systems.
- **Propagation can amplify errors:** Incorrect initial risk scores can spread through legitimate financial relationships. Thresholds, decay, and propagation limits are necessary safeguards.
- **Audit context matters:** Legitimate business processes can generate repeated amounts, period-end adjustments, reversals, and unusual account combinations.
- **Explainability is essential:** Every flagged entry should retain the underlying signals and supporting transaction references.

---

## Long-Term Vision

The project will evolve through three stages:

1. **Beta — Detect:** Identify unusual journal entries using independent forensic accounting and statistical tests.
2. **Version 1.0 — Connect:** Model relationships between financial entities and examine how risk signals propagate through an organization.
3. **Version 2.0 — Learn:** Train a neural network to estimate fraud probabilities from historical journal-entry patterns and compare its performance with the rule-based and graph-based approaches.

The broader goal is to investigate whether combining traditional forensic accounting, graph-based risk propagation, and machine learning can improve the prioritization of journal entries for audit review.

This project is intended as a portfolio demonstration of Python development, data analytics, forensic accounting concepts, and technology-enabled risk assessment relevant to Internal Audit, IT Assurance, and SOX compliance.