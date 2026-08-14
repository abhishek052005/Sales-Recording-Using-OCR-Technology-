from typing import Optional
from pydantic import BaseModel


class Party(BaseModel):
    name: Optional[str] = None
    gstin: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None


class InvoiceItem(BaseModel):
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    tax_rate: Optional[float] = None
    amount: Optional[float] = None


class Invoice(BaseModel):
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None

    vendor: Party
    customer: Party

    items: list[InvoiceItem]

    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None

    currency: Optional[str] = None
