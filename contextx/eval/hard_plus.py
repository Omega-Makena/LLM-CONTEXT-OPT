"""A larger, difficulty-stratified eval set of confusable clusters.

Each cluster is a set of docs that share heavy vocabulary and differ by one
distinguishing detail (the option greeks, liquidity/leverage ratios, order types,
risk measures, bond types, payment rails, inventory methods, margin types). A
bi-encoder tends to confuse siblings; a cross-encoder reads query+doc jointly and
resolves the distinguishing detail — so the set is enriched toward queries where
reranking matters.

Financial domain (Innova-relevant), public knowledge.
"""

from __future__ import annotations

from ..types import Document, Source
from .dataset import EvalExample


def _d(doc_id: str, text: str) -> Document:
    return Document(text=text, doc_id=doc_id, source=Source.KNOWLEDGE_BASE,
                    metadata={"domain": "finance"})


HARD_PLUS_CORPUS: list[Document] = [
    # option greeks
    _d("delta", "Delta measures an option's price sensitivity to a $1 change in the "
                "underlying asset's price."),
    _d("gamma", "Gamma measures the rate of change of an option's delta as the "
                "underlying price moves."),
    _d("theta", "Theta measures an option's value lost per day from time decay, all "
                "else equal."),
    _d("vega", "Vega measures an option's price sensitivity to a one-point change in "
               "the underlying's implied volatility."),
    _d("rho", "Rho measures an option's price sensitivity to a one-point change in "
              "interest rates."),
    # liquidity ratios
    _d("current_ratio", "The current ratio is current assets divided by current "
                        "liabilities, a broad liquidity measure."),
    _d("quick_ratio", "The quick ratio, or acid-test, is current assets minus "
                      "inventory over current liabilities; it excludes inventory."),
    _d("cash_ratio", "The cash ratio is cash and equivalents over current "
                     "liabilities, the strictest liquidity measure."),
    # leverage / coverage
    _d("debt_to_equity", "Debt-to-equity is total debt divided by shareholders' "
                         "equity, a leverage measure."),
    _d("interest_coverage", "Interest coverage is EBIT divided by interest expense: "
                            "the ability to pay interest from operating earnings."),
    _d("debt_to_ebitda", "Debt-to-EBITDA is total debt divided by EBITDA: leverage "
                         "relative to earnings."),
    # order types
    _d("market_order", "A market order executes immediately at the best available "
                       "current price."),
    _d("limit_order", "A limit order executes only at a specified price or better."),
    _d("stop_order", "A stop order becomes a market order once the price reaches a "
                     "trigger level."),
    _d("stop_limit_order", "A stop-limit order becomes a limit order once the trigger "
                           "price is hit."),
    # risk measures
    _d("var", "Value at Risk (VaR) estimates the maximum loss over a period at a "
              "given confidence level."),
    _d("cvar", "Conditional VaR, or expected shortfall, is the average loss beyond "
               "the VaR threshold."),
    _d("beta", "Beta measures a security's volatility relative to the overall "
               "market."),
    _d("sharpe", "The Sharpe ratio is excess return per unit of total volatility, a "
                 "risk-adjusted return."),
    # bond types
    _d("zero_coupon", "A zero-coupon bond pays no periodic interest and is issued at "
                      "a discount to face value."),
    _d("callable_bond", "A callable bond lets the issuer redeem it before maturity."),
    _d("convertible_bond", "A convertible bond can be exchanged for a fixed number of "
                           "the issuer's shares."),
    _d("floating_rate", "A floating-rate note pays interest that resets periodically "
                        "to a reference rate."),
    # payment rails
    _d("ach_p", "ACH transfers are batched and low-cost and settle in one to two "
                "business days."),
    _d("wire_p", "A wire transfer settles same-day and is irrevocable once sent."),
    _d("rtp_p", "RTP real-time payments settle instantly, 24/7, with immediate "
                "finality."),
    _d("card_p", "Card payments authorize instantly but settle in batches days "
                 "later."),
    # inventory accounting
    _d("fifo", "FIFO expenses the oldest inventory costs first."),
    _d("lifo", "LIFO expenses the newest inventory costs first."),
    _d("wavg", "Weighted-average costing expenses inventory at the average unit "
               "cost."),
    # margin types
    _d("initial_margin", "Initial margin is the collateral required to open a "
                         "leveraged position."),
    _d("maintenance_margin", "Maintenance margin is the minimum equity to keep a "
                             "position open before a margin call."),
    _d("variation_margin", "Variation margin is the daily cash settlement of gains "
                           "and losses on a position."),
    # market data
    _d("bid", "The bid is the highest price a buyer will pay."),
    _d("ask", "The ask is the lowest price a seller will accept."),
    _d("spread", "The bid-ask spread is the difference between the ask and the bid."),
]


def _q(query: str, relevant: str) -> EvalExample:
    return EvalExample(query, [relevant])


HARD_PLUS_QUERIES: list[EvalExample] = [
    _q("which greek measures the time decay of an option per day?", "theta"),
    _q("an option's price sensitivity to implied volatility", "vega"),
    _q("the rate of change of delta as the underlying moves", "gamma"),
    _q("an option's sensitivity to a change in interest rates", "rho"),
    _q("an option's sensitivity to the underlying asset's price", "delta"),
    _q("liquidity ratio that excludes inventory from current assets", "quick_ratio"),
    _q("the strictest liquidity ratio, counting only cash", "cash_ratio"),
    _q("current assets divided by current liabilities", "current_ratio"),
    _q("leverage measured relative to EBITDA earnings", "debt_to_ebitda"),
    _q("ability to pay interest from operating earnings", "interest_coverage"),
    _q("total debt relative to shareholders' equity", "debt_to_equity"),
    _q("order that executes only at a specified price or better", "limit_order"),
    _q("order that becomes a market order at a trigger price", "stop_order"),
    _q("order that becomes a limit order once a trigger is hit", "stop_limit_order"),
    _q("order that fills immediately at the best available price", "market_order"),
    _q("average loss beyond the value-at-risk threshold", "cvar"),
    _q("maximum loss over a period at a confidence level", "var"),
    _q("a security's volatility relative to the market", "beta"),
    _q("risk-adjusted return per unit of total volatility", "sharpe"),
    _q("bond paying no periodic interest, issued at a discount", "zero_coupon"),
    _q("bond the issuer can redeem before maturity", "callable_bond"),
    _q("bond exchangeable for a fixed number of shares", "convertible_bond"),
    _q("note whose interest resets to a reference rate", "floating_rate"),
    _q("same-day irrevocable bank transfer", "wire_p"),
    _q("instant 24/7 payment with immediate finality", "rtp_p"),
    _q("batched low-cost transfer settling in one to two days", "ach_p"),
    _q("payment that authorizes instantly but settles in batches later", "card_p"),
    _q("expensing the oldest inventory costs first", "fifo"),
    _q("expensing the newest inventory costs first", "lifo"),
    _q("inventory expensed at the average unit cost", "wavg"),
    _q("collateral required to open a leveraged position", "initial_margin"),
    _q("minimum equity to hold a position before a margin call", "maintenance_margin"),
    _q("daily cash settlement of a position's gains and losses", "variation_margin"),
    _q("the highest price a buyer will pay", "bid"),
    _q("the lowest price a seller will accept", "ask"),
    _q("difference between the ask and the bid", "spread"),
]
