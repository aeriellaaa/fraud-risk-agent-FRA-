from app.models import IEEETransactionIn
from app.agents.scoring_agent_ieee import score_transaction_ieee

txn = IEEETransactionIn(
    transaction_id="test_001", TransactionAmt=500.0, ProductCD="W",
    card1=13553, card2=225.0, card4="visa", card6="debit",
    addr1=299.0, P_emaildomain="gmail.com", C13=50.0, C1=10.0
)
result = score_transaction_ieee(txn)
print(f"Score: {result.score}")
for e in result.evidence:
    print(f"  {e.signal}: {e.description}")
