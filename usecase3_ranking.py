"""
usecase3_ranking.py — Generate ranked predictors table for Use Case 3.
"""

import re
import pandas as pd
from utils.db import load_all_findings


def extract_metric(effect_size: str, performance: str) -> tuple:
    """Extract best numeric metric from effect_size or performance strings."""
    text = f"{effect_size or ''} {performance or ''}".lower()
    
    auc = re.search(r'auc[=\s]*([0-9]\.[0-9]+)', text)
    if auc:
        return "AUC", float(auc.group(1))
    
    cidx = re.search(r'c-index[=\s]*([0-9]\.[0-9]+)', text)
    if cidx:
        return "C-index", float(cidx.group(1))
    
    or_match = re.search(r'\bor[=\s]*([0-9]+\.?[0-9]*)', text)
    if or_match:
        return "OR", float(or_match.group(1))
    
    hr_match = re.search(r'\bhr[=\s]*([0-9]+\.?[0-9]*)', text)
    if hr_match:
        return "HR", float(hr_match.group(1))
    
    return None, None


def generate_ranking() -> pd.DataFrame:
    df = load_all_findings()
    
    if df.empty:
        return pd.DataFrame()
    
    mortality_df = df[
        df["outcome"].str.contains("mortality|death|survival",
                                    case=False, na=False)
    ].copy()
    
    rows = []
    for _, row in mortality_df.iterrows():
        metric_name, metric_value = extract_metric(
            row.get("effect_size"),
            row.get("performance")
        )
        if metric_value is None:
            continue
        
        rows.append({
            "predictor": row.get("predictor"),
            "best_metric": metric_name,
            "value": metric_value,
            "effect_size": row.get("effect_size"),
            "study": row.get("paper_id"),
            "outcome": row.get("outcome"),
            "method": row.get("method"),
            "source_sentence": row.get("source_sentence"),
        })
    
    if not rows:
        return pd.DataFrame()
    
    ranked = pd.DataFrame(rows)
    ranked = ranked.sort_values("value", ascending=False)
    ranked = ranked.drop_duplicates(subset=["predictor"], keep="first")
    return ranked.reset_index(drop=True)


if __name__ == "__main__":
    df = generate_ranking()
    if df.empty:
        print("No ranked findings found")
    else:
        print(f"Ranked {len(df)} predictors")
        print(df[["predictor", "best_metric", "value", "study"]].to_string())
        df.to_csv("usecase3_ranked_predictors.csv", index=False)
        print("Saved to usecase3_ranked_predictors.csv")