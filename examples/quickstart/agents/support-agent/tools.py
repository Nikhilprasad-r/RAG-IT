"""Tools for the support agent.

Any function listed under `tools.local` in agent.yaml is given to the model.
The docstring becomes the tool description, so keep it accurate.
"""


def lookup_order(order_id: str) -> str:
    """Look up an order by id and return its status."""
    # Replace with a call to your order system.
    return f"Order {order_id} is shipped."
