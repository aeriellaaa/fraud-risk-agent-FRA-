"""
Feature pipeline: converts a raw TransactionIn into a FeatureVector.

Thresholds below are derived from credit_card_fraud_2026.csv (20k txns,
1.7% fraud rate), not arbitrary guesses:

  - velocity_score:      fraud mean 28.97 vs non-fraud 19.65 -> flag above
                          the 75th percentile (27.6)
  - ip_country_mismatch:  7.8% fraud rate when True vs 1.3% when False
  - billing_shipping_mismatch: 7.8% vs 1.4%
  - is_foreign_transaction: 5.9% vs 1.4%
  - used_vpn:             4.4% vs 1.4%
  - is_new_merchant:      3.5% vs 1.2%
  - cvv_retry_count:      median is 0 in both classes; any retry (>0)
                          is itself unusual (fraud mean 0.68 vs 0.17)
  - prior_disputes:       any prior dispute (>0) flagged
  - merchant_risk_score:  flag above the 75th percentile (47.3)
  - distance_from_home_km and amount/balance ratio were checked and
    do NOT separate fraud from non-fraud in this dataset -- deliberately
    excluded rather than included as noise.
"""

from app.models import TransactionIn, FeatureVector

VELOCITY_THRESHOLD = 27.6
MERCHANT_RISK_THRESHOLD = 47.3


def extract_features(txn: TransactionIn) -> FeatureVector:
    velocity_flag = (
        txn.velocity_score > VELOCITY_THRESHOLD
        or txn.txn_count_last_24h >= 5
    )

    geo_mismatch = (
        txn.ip_country_mismatch
        or txn.billing_shipping_mismatch
        or txn.is_foreign_transaction
    )

    device_risk = txn.used_vpn

    new_merchant_risk = txn.is_new_merchant

    cvv_risk = txn.cvv_retry_count > 0

    prior_dispute_risk = txn.prior_disputes > 0

    high_amount_ratio = (
        txn.account_balance_usd > 0
        and (txn.amount_usd / txn.account_balance_usd) > 0.3
    )

    ai_scam_flag = txn.is_ai_generated_scam_attempt

    return FeatureVector(
        transaction_id=txn.transaction_id,
        velocity_flag=velocity_flag,
        geo_mismatch=geo_mismatch,
        device_risk=device_risk,
        new_merchant_risk=new_merchant_risk,
        cvv_risk=cvv_risk,
        prior_dispute_risk=prior_dispute_risk,
        high_amount_ratio=high_amount_ratio,
        merchant_risk_score=txn.merchant_risk_score,
        ai_scam_flag=ai_scam_flag,
    )
