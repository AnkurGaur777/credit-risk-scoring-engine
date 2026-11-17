"""Regenerate every written deliverable from reports/metrics_*.json.  Usage: python scripts/build_docs.py"""

from credit_risk import config as C
from credit_risk import reporting as R

C.ensure_dirs()
ms = R.load()
if "A" not in ms:
    raise SystemExit(
        "Run `make run-a` (and optionally `make run-b`) first - reports/metrics_A.json not found."
    )
R.write_data_audit(ms["A"], C.DOCS / "data_audit.md")
R.write_feature_dictionary(C.DOCS / "feature_dictionary.md")
R.write_monitoring_plan(C.DOCS / "monitoring_plan.md")
R.write_results(ms, C.REPORTS / "results.md")
R.write_model_card(ms, C.DOCS / "model_card.md")
R.draw_architecture(C.DOCS / "architecture.png")
R.write_business_summary_pdf(ms, C.DOCS / "business_summary.pdf")
print(
    "wrote docs/: data_audit.md, feature_dictionary.md, monitoring_plan.md, model_card.md, architecture.png, business_summary.pdf; reports/results.md"
)
