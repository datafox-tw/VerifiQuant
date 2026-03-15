from __future__ import annotations

import json
from typing import Dict, List


TAXONOMY: Dict[str, List[str]] = {
    "accounting_and_financial_reporting": [
        "balance_sheets",
        "cash_flow_statements",
        "depreciation_and_amortization",
        "income_statements",
    ],
    "corporate_finance": [
        "capital_structure",
        "div_policies",
        "eps",
        "misc",
        "wacc",
    ],
    "crypto_finance": [
        "defi",
        "global_events_impact",
        "regulations",
        "sentiments",
        "tokenomics",
        "transaction_analysis",
        "whale_detection",
    ],
    "finance_regulation": [
        "aml",
        "basel",
        "compliance",
        "kyc",
        "security",
    ],
    "financial_markets": [
        "bond_pricing",
        "forex_trading",
        "option_pricing",
        "stock_market_analysis",
    ],
    "financial_ratios": [
        "effratio",
        "levratio",
        "liqratio",
        "profitratio",
    ],
    "fintech": [
        "digital_banking",
        "payment_technologies",
        "robo_advisors",
    ],
    "investment_analysis": [
        "ci",
        "irr",
        "npv",
        "rar",
        "roi",
    ],
    "mergers_and_acquisitions": [
        "deal_structure",
        "due_diligence",
        "post_merger_integration",
        "synergies_and_cost_savings",
        "valuation_methods",
    ],
    "personal_finance": [
        "budgeting",
        "loanrepay",
        "personalinvest",
        "saveretire",
        "taxcalc",
    ],
    "risk_management": [
        "regulatory_compliance",
        "risk_appetite",
        "risk_metrics",
        "scenario_planning",
        "sensitivity_analysis",
        "stress_testing",
        "var",
    ],
    "sustainable_finance": [
        "carbon_credits",
        "esg_investing",
        "green_bonds",
        "social_impact_investing",
        "sustainability_reporting",
    ],
}


def taxonomy_json(indent: int = 2) -> str:
    return json.dumps(TAXONOMY, ensure_ascii=False, indent=indent, sort_keys=True)


def is_valid_domain_topic(domain: str, topic: str) -> bool:
    return domain in TAXONOMY and topic in TAXONOMY[domain]

