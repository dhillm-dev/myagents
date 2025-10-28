"""
Market data connectors package

Note: Avoid importing heavy optional deps (e.g., CCXT/pandas) at package import time.
Import submodules directly where needed to prevent startup failures when optional
dependencies are not installed.
"""

# Intentionally avoid eager imports here to prevent optional dependency issues.
# Import submodules explicitly in callers:
#   from .connectors.market_yf import get_yf_connector
#   from .connectors.market_ccxt import get_ccxt_connector

__all__ = [
    'market_yf',
    'market_ccxt'
]