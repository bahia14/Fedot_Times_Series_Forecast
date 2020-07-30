from core.chain.chain_validation import validate
from core.chain.chain import Chain


def constraint_function(chain: Chain):
    try:
        validate(chain)
        return True
    except ValueError:
        return False
