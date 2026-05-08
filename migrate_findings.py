"""
migrate_findings.py — Migrate prog_findings from papers table into findings table.
Run once to populate the findings table from existing data.

Usage:
    python migrate_findings.py
"""

import json
from sqlalchemy import create_engine, text
import config
from utils.logger import get_logger

logger = get_logger(__name__)

def migrate():
    engine = create_engine(f"sqlite:///{config.DB_PATH}", echo=False)
    
    with engine.connect() as conn:
        # Load all papers with prog_findings
        rows = conn.execute(text(
            "SELECT paper_id, prog_findings FROM papers WHERE prog_findings IS NOT NULL"
        )).fetchall()
        
        migrated = 0
        for paper_id, prog_findings_raw in rows:
            try:
                findings = json.loads(prog_findings_raw)
            except Exception:
                continue
            
            # Delete existing findings for this paper
            conn.execute(text(
                "DELETE FROM findings WHERE paper_id = :pid"
            ), {"pid": paper_id})
            
            # Insert each finding
            for idx, f in enumerate(findings):
                conn.execute(text("""
                    INSERT INTO findings (
                        finding_id, paper_id, predictor, outcome,
                        timing, method, effect_size, performance,
                        notes, source_sentence, confidence
                    ) VALUES (
                        :finding_id, :paper_id, :predictor, :outcome,
                        :timing, :method, :effect_size, :performance,
                        :notes, :source_sentence, :confidence
                    )
                """), {
                    "finding_id": f"{paper_id}_f{idx:03d}",
                    "paper_id": paper_id,
                    "predictor": f.get("predictor"),
                    "outcome": f.get("outcome"),
                    "timing": f.get("timing"),
                    "method": f.get("method"),
                    "effect_size": f.get("effect_size"),
                    "performance": f.get("performance"),
                    "notes": f.get("notes"),
                    "source_sentence": f.get("source_sentence"),
                    "confidence": float(f.get("confidence", 0.0)),
                })
                migrated += 1
        
        conn.commit()
        print(f"✓ Migrated {migrated} findings from {len(rows)} papers")

if __name__ == "__main__":
    migrate()