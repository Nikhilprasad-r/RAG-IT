"""Tools for the support agent.

Any function listed under `tools.local` in agent.yaml is given to the model.
The docstring becomes the tool description, so keep it accurate.
"""


def lookup_order(order_id: str) -> str:
    """Look up an order by id and return its status."""
    # Replace with a call to your order system.
    return f"Order {order_id} is shipped."


def issue_refund(order_id: str, amount: float, reason: str) -> str:
    """Refund part or all of an order. Requires a human's approval before it runs."""
    # Replace with a call to your payments system.
    return f"Refunded {amount:.2f} on order {order_id} ({reason})."
