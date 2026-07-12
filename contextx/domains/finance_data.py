"""Finance eval set — a labeled benchmark across capital markets, payments,
lending, and compliance/KYC-AML. Public, generic financial knowledge.

Queries are written as finance-specific lexical traps (CUSIP vs ISIN, KYC vs
AML, IBAN vs SWIFT vs routing, EPS vs P/E, DTI vs LTV, 10-K vs 10-Q, SAR vs
CTR) — the near-synonym confusions where a bi-encoder is fooled by surface
overlap and the hybrid lexical channel + reranker earn their keep.

Swap this for your proprietary corpus/labels via `contextx.eval.load_jsonl`.
"""

from __future__ import annotations

from ..types import Document, Source
from ..eval.dataset import EvalExample


def _d(doc_id: str, text: str) -> Document:
    return Document(text=text, doc_id=doc_id, source=Source.KNOWLEDGE_BASE,
                    metadata={"domain": "finance"})


FINANCE_CORPUS: list[Document] = [
    # --- capital markets ---------------------------------------------------
    _d("ticker", "A ticker symbol is a short code identifying a publicly traded "
                 "security, for example AAPL for Apple."),
    _d("cusip", "A CUSIP is a 9-character alphanumeric code that identifies a North "
                "American security."),
    _d("isin", "An ISIN is a 12-character international securities identifier; it "
               "embeds a two-letter country code and a check digit."),
    _d("tenk", "A 10-K is a company's annual report filed with the SEC, containing "
               "audited financial statements and risk factors."),
    _d("tenq", "A 10-Q is a company's quarterly report filed with the SEC, with "
               "unaudited interim financial statements."),
    _d("eps", "Earnings per share (EPS) is net income divided by the number of "
              "shares outstanding."),
    _d("pe_ratio", "The price-to-earnings (P/E) ratio is the share price divided by "
                   "earnings per share."),
    _d("market_cap", "Market capitalization is the share price multiplied by the "
                     "number of shares outstanding."),
    _d("settlement_t1", "US equity trades settle on a T+1 basis: one business day "
                        "after the trade date."),
    _d("bid_ask", "The bid-ask spread is the difference between the highest bid "
                  "price and the lowest ask price."),
    _d("ytm", "Yield to maturity is the total return anticipated on a bond if it is "
              "held until it matures."),
    _d("coupon", "A bond's coupon is the fixed periodic interest payment made on its "
                 "face value."),
    _d("etf", "An ETF (exchange-traded fund) is a pooled investment fund that trades "
              "on an exchange like a stock."),
    _d("short_sale", "Short selling is borrowing a security to sell it, intending to "
                     "buy it back later at a lower price."),
    # --- payments / banking ------------------------------------------------
    _d("iban", "An IBAN (International Bank Account Number) identifies a specific "
               "bank account across borders."),
    _d("swift_bic", "A SWIFT/BIC code identifies a specific bank in international "
                    "wire transfers."),
    _d("routing", "A US ABA routing number is a 9-digit code identifying a bank for "
                  "domestic transfers."),
    _d("ach", "ACH is a US electronic network for batched bank-to-bank transfers "
              "such as payroll and direct debits."),
    _d("interchange", "Interchange is the fee an acquiring bank pays the card issuer "
                      "on each card transaction."),
    _d("chargeback", "A chargeback is a forced reversal of a card payment initiated "
                     "by the cardholder's issuing bank."),
    _d("pci_dss", "PCI DSS is the security standard governing the handling of "
                  "cardholder data."),
    _d("three_ds", "3-D Secure adds a cardholder authentication step to card-not-"
                   "present transactions to reduce fraud."),
    _d("settlement_pay", "Payment settlement is the transfer of funds between banks "
                         "that finalizes a transaction."),
    # --- lending / credit --------------------------------------------------
    _d("credit_score", "A credit score such as FICO summarizes a borrower's "
                       "creditworthiness on a scale from 300 to 850."),
    _d("apr", "APR (annual percentage rate) is the annualized cost of a loan "
              "including interest and fees."),
    _d("dti", "The debt-to-income (DTI) ratio is total monthly debt payments divided "
              "by gross monthly income."),
    _d("ltv", "The loan-to-value (LTV) ratio is the loan amount divided by the value "
              "of the collateral."),
    _d("collateral", "Collateral is an asset a borrower pledges to secure a loan, "
                     "which the lender can seize on default."),
    _d("underwriting", "Underwriting is the process of assessing a borrower's risk to "
                       "decide whether to extend credit and on what terms."),
    _d("amortization", "Amortization is repaying a loan through scheduled payments "
                       "that cover both principal and interest."),
    _d("default_loan", "A default occurs when a borrower fails to meet the repayment "
                       "terms of a loan."),
    # --- compliance / KYC-AML ---------------------------------------------
    _d("kyc", "KYC (Know Your Customer) is verifying a customer's identity before "
              "and during onboarding."),
    _d("aml", "AML (Anti-Money-Laundering) rules require firms to detect and report "
              "suspicious financial activity."),
    _d("sar", "A SAR (Suspicious Activity Report) is filed with regulators when a "
              "firm suspects illicit activity."),
    _d("ctr", "A CTR (Currency Transaction Report) is filed for cash transactions "
              "above a regulatory threshold."),
    _d("sanctions", "Sanctions screening checks parties against government watchlists "
                    "such as OFAC's."),
    _d("pep", "A PEP (Politically Exposed Person) carries elevated AML risk and "
              "requires enhanced due diligence."),
    _d("mifid", "MiFID II is an EU regulation governing investment services and "
                "market transparency."),
    _d("basel", "The Basel III framework sets bank capital and liquidity "
                "requirements to strengthen the banking system."),
    # --- distractors -------------------------------------------------------
    _d("noise_coffee", "The office espresso machine needs descaling every month."),
    _d("noise_parking", "Visitor parking is on level 2; register at reception."),
]


FINANCE_QUERIES: list[EvalExample] = [
    # capital markets (traps: cusip/isin, 10k/10q, eps/pe)
    EvalExample("what nine-character code identifies a North American security?",
                ["cusip"], "trap: isin (12-char)"),
    EvalExample("the twelve-character international securities identifier",
                ["isin"], "trap: cusip"),
    EvalExample("annual audited report a company files with the SEC",
                ["tenk"], "trap: tenq"),
    EvalExample("unaudited quarterly report filed with the SEC",
                ["tenq"], "trap: tenk"),
    EvalExample("share price divided by earnings per share",
                ["pe_ratio"], "trap: eps"),
    EvalExample("net income divided by shares outstanding",
                ["eps"], "trap: pe_ratio, market_cap"),
    EvalExample("how many business days until a US stock trade settles?",
                ["settlement_t1"], ""),
    EvalExample("difference between the highest bid and the lowest ask",
                ["bid_ask"], ""),
    EvalExample("total return if you hold a bond until it matures",
                ["ytm"], "trap: coupon"),
    # payments (traps: iban/swift/routing, interchange/chargeback)
    EvalExample("cross-border identifier for a specific bank account",
                ["iban"], "trap: swift_bic, routing"),
    EvalExample("code identifying a bank in an international wire transfer",
                ["swift_bic"], "trap: iban, routing"),
    EvalExample("nine-digit US code identifying a bank for domestic transfers",
                ["routing"], "trap: swift_bic, cusip"),
    EvalExample("forced reversal of a card payment by the cardholder's bank",
                ["chargeback"], "trap: interchange"),
    EvalExample("fee an acquirer pays the issuer on each card transaction",
                ["interchange"], "trap: chargeback"),
    EvalExample("security standard for handling cardholder data",
                ["pci_dss"], ""),
    # lending (traps: dti/ltv, apr/credit_score)
    EvalExample("annualized cost of a loan including its fees",
                ["apr"], "trap: credit_score"),
    EvalExample("monthly debt payments divided by gross monthly income",
                ["dti"], "trap: ltv"),
    EvalExample("loan amount divided by the value of the collateral",
                ["ltv"], "trap: dti"),
    EvalExample("asset a borrower pledges to secure a loan",
                ["collateral"], ""),
    EvalExample("score from 300 to 850 summarizing creditworthiness",
                ["credit_score"], "trap: apr"),
    # compliance (traps: kyc/aml, sar/ctr, mifid/basel)
    EvalExample("verifying a customer's identity at onboarding",
                ["kyc"], "trap: aml"),
    EvalExample("rules requiring firms to detect and report suspicious activity",
                ["aml"], "trap: kyc"),
    EvalExample("report filed when a firm suspects illicit activity",
                ["sar"], "trap: ctr"),
    EvalExample("report filed for large cash transactions above a threshold",
                ["ctr"], "trap: sar"),
    EvalExample("checking parties against OFAC watchlists",
                ["sanctions"], ""),
    EvalExample("high-risk customer requiring enhanced due diligence",
                ["pep"], ""),
    EvalExample("EU regulation for investment services and market transparency",
                ["mifid"], "trap: basel"),
    EvalExample("framework setting bank capital and liquidity requirements",
                ["basel"], "trap: mifid"),
]
