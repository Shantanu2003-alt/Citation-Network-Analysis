# Citation-Network-Analysis
This project builds a smart system to predict future academic collaborations and rank research papers more accurately than just counting citations. It uses network features, time-based importance, and the quality of citations to give a deeper, more complete view of academic impact.
# Graph-Based Link Prediction & Academic Impact Ranking
A comprehensive machine learning framework for analyzing academic networks through link prediction and multi-dimensional impact ranking. This project implements advanced algorithms to predict missing connections in citation and collaboration networks while providing sophisticated paper impact assessment beyond traditional citation counts.
# Project Overview
This repository contains two main components:
Link Prediction System: Predicts potential future citations and collaborations using graph-based features and machine learning
Impact Ranking System: Evaluates academic paper influence using multiple weighted metrics including temporal factors, citation quality, and network authority
# Features
Link Prediction
Multiple ML Algorithms: Decision Trees and Logistic Regression implementations
Graph Feature Engineering: Common neighbors, preferential attachment, and degree-based features
Cross-Validation: Robust model evaluation with 5-fold cross-validation
Comprehensive Metrics: AUC, F1-score, precision, recall, and confusion matrices
Dual Network Support: Citation networks (directed) and collaboration networks (undirected)
Impact Ranking
Multi-Metric Scoring: Combines 4 different influence measures
Temporal Weighting: Recent citations weighted more heavily
Citation Quality Assessment: Considers the influence of citing papers
Network Authority: HITS/SALSA algorithm for structural importance
Visual Analytics: Comprehensive plotting and correlation analysis
Performance Evaluation: MAP@10 and Spearman correlation metrics
# Technology Stack
Core: Python 3.7+, NetworkX, NumPy, Pandas
Machine Learning: Scikit-learn
Visualization: Matplotlib
Statistical Analysis: SciPy
Data Handling: Gzip compression support
# Datasets
The project works with standard academic network datasets from the snap stanford website:
Citation Networks: Cit-HepPh.txt (High Energy Physics citations)
Collaboration Networks: ca-HepPh.txt.gz (Co-authorship networks)
Multi-domain Support: ca-AstroPh.txt.gz, cit-HepTh.txt.gz
# Installation
Clone the repository
bashgit clone <repository-url>
cd link-prediction-impact-ranking
# Install dependencies
bashpip install numpy pandas networkx scikit-learn matplotlib scipy
Prepare datasets
Download the required network datasets
Place them in the project root directory
Ensure proper file naming as referenced in the code
# Usage
Link Prediction
Decision Trees Approach:
pythonpython link_prediction_decision_trees.py
Logistic Regression Approach:
pythonpython link_prediction_logistic_regression.py
# Expected Output:
Cross-validated AUC scores
Test performance metrics
Confusion matrices for both citation and collaboration networks
Impact Ranking
pythonpython impact_ranking.py
Features:
Generates comprehensive paper rankings
Creates visualization plots for top papers
Outputs correlation analysis and MAP@10 scores
Compares traditional citation-based vs. multi-metric ranking
# Methodology
Link Prediction Features
Common Neighbors: Number of mutual connections between node pairs
Preferential Attachment: Product of node degrees (rich-get-richer effect)
# Impact Ranking Components
Citation Count (α=0.35): Basic influence measure
Time-Weighted Score (β=0.35): Recent citations weighted by recency
Quality Score (γ=0.20): Weighted by citing paper importance
SALSA Rank (δ=0.10): Network structural authority
# Final Rank Formula:
Final Rank = α×Norm(Citations) + β×Norm(Time-Weight) + γ×Norm(Quality) + δ×Norm(SALSA)
# Results & Evaluation
Link Prediction Performance
AUC Scores: Typically 0.85-0.95 for both algorithms
Cross-Validation: 5-fold CV for robust performance assessment
Network Comparison: Citation vs. collaboration network analysis
Impact Ranking Validation
Spearman Correlation: Measures rank correlation with traditional citation counts
Mean Average Precision (MAP@10): Evaluates top-paper prediction quality
Visual Analysis: Comparative bar charts for different ranking components
# 📁 Project Structure
link-prediction-impact-ranking/
├── link_prediction_decision_trees.py    # Decision tree implementation
├── link_prediction_logistic_regression.py # Logistic regression implementation  
├── impact_ranking.py                     # Multi-metric impact ranking
├── data/                                 # Dataset directory
│   ├── Cit-HepPh.txt                    # Citation network data
│   ├── ca-HepPh.txt.gz                  # Collaboration network data
│   └── ...                              # Additional datasets
├── results/                              # Output plots and analysis
└── README.md                            # Project documentation
# Research Applications
Academic Network Analysis: Understanding citation and collaboration patterns
Research Impact Assessment: Beyond simple citation counting
Recommendation Systems: Suggesting potential collaborations or relevant papers
Network Evolution: Predicting future academic network structures
Research Policy: Data-driven insights for funding and evaluation decisions
# Sample Output (Not the actual output)
--- Training Citation network dataset using Decision Tree ---
Cross-Validated AUC: 0.8924
Test AUC: 0.8876
Accuracy: 0.8234
Precision: 0.8156
Recall: 0.8298
F1 Score: 0.8226
# Final Paper Ranking Sample 
Top 10 Papers Based on Final Rank:
PaperID  Citation Count  Time-Weighted Score  Quality Score  SALSA Rank  Final Rank
  12345             156                 8.45           0.89        0.012        0.8234
# Contributing
Fork the repository
Create a feature branch (git checkout -b feature/enhancement)
Implement your changes with appropriate tests
Commit changes (git commit -m 'Add feature: description')
Push to branch (git push origin feature/enhancement)
Create a Pull Request
# License
This project is licensed under the MIT License - see the LICENSE file for details.
# Important Notes
Dataset Requirements: Ensure datasets are in the correct format (space-separated node pairs)
Memory Usage: Large networks may require significant RAM for feature computation
Random Seed: Set to 42 for reproducible results across runs
File Formats: Supports both compressed (.gz) and uncompressed text files
