class RegulatoryAuditTools:
    def __init__(self, vector_db):
        self.vector_db = vector_db

    def check_psi_threshold(self, psi_value: float) -> str:
        if psi_value < 0.10:
            return f"PSI = {psi_value}: Stable. No population drift detected."
        elif psi_value < 0.25:
            return f"PSI = {psi_value}: Moderate shift. Requires targeted feature drift monitoring."
        else:
            return f"PSI = {psi_value}: Critical Alert. Significant drift; triggers mandatory retraining."

    def check_fair_lending(self, disparate_impact_ratio: float) -> str:
        if disparate_impact_ratio >= 0.80:
            return f"Disparate Impact Ratio = {disparate_impact_ratio:.2f}: Compliant with 4/5ths (80%) standard."
        else:
            return f"Disparate Impact Ratio = {disparate_impact_ratio:.2f}: Non-Compliant. Adverse disparate impact flagged."

    def retrieve_regulatory_clause(self, query: str) -> str:
        return self.vector_db.query(query)
