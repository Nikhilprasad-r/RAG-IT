"""The structure the extractor returns. Replace `Invoice` with your own model.

Field descriptions are shown to the model, so say what each field means and
how to format it. Make a field optional when documents may not contain it.
"""

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str
    quantity: float = Field(description="Number of units; 1 if not stated.")
    unit_price: float | None = Field(default=None, description="Price per unit, no currency sign.")
    amount: float = Field(description="Line total, no currency sign.")


class Invoice(BaseModel):
    vendor: str = Field(description="Company that issued the invoice.")
    invoice_number: str | None = None
    invoice_date: str | None = Field(default=None, description="ISO 8601 date, e.g. 2026-10-30.")
    due_date: str | None = Field(default=None, description="ISO 8601 date.")
    currency: str | None = Field(default=None, description="ISO 4217 code, e.g. EUR or USD.")
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: float | None = None
    tax: float | None = None
    total: float = Field(description="Amount due, no currency sign.")
