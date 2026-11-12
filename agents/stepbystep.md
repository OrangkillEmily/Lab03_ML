# Step by Step

```text
customer_segmentation.csv
        |
        v
      EDA --------------------> docs/images + outlier_summary.csv
        |
        v
  Preprocessing --------------> customers_clean.csv
        |                       X_scaled.csv + scaler.json
        v
     Training ----------------> best_kmeans.pkl + experiment_results.csv
        |
        v
    Evaluation ---------------> customer_clusters.csv
        |                       cluster_profile.csv + figures
        v
 Experiments / Convergence ---> grid_search + feature comparison + history
        |
        v
   Visualization -------------> reports/figures/<Run_ID>/
        |
        v
      Report -----------------> report_<Run_ID>.md/.html/.pdf

New customer -> saved scaler -> KMeans.predict -> Cluster
                                      |
                                      v
                           reports/inference_history.csv

Every stage -> progress_results/<Run_ID>.json + logs/<Run_ID>.log
```
