import re


def find_first_match(patterns, text):
    """Return the first matching group from a list of regex patterns."""
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def extract_invoice_data(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    data = {
        "vendor_name": "",
        "company_name": "",
        "address": "",
        "phone": "",
        "invoice_number": "",
        "invoice_date": "",
        "invoice_time": "",
        "cashier": "",
        "gst_number": "",
        "total_amount": "",
        "cash_received": "",
        "change": "",
        "items": [],
    }

    if not lines:
        return data

    # ----------------------------
    # Vendor / Company
    # ----------------------------
    data["vendor_name"] = lines[0] if len(lines) > 0 else ""
    data["company_name"] = lines[1] if len(lines) > 1 else ""

    # ----------------------------
    # Address
    # ----------------------------
    address_lines = []
    for line in lines[2:8]:
        # Stop at phone, Cash Bill, or Date
        if (
            re.search(r"\d{2,4}[-\s]?\d{3,4}", line)
            or "Cash Bill" in line
            or "Date" in line
        ):
            break
        address_lines.append(line)

    data["address"] = ", ".join(address_lines)

    # ----------------------------
    # Phone
    # ----------------------------
    phone_patterns = [
        r"(\d{2,4}\s*[-\s]\s*\d{3,4}\s*\d{3,4})",  # Handles 07-355 2616
        r"(\d{2,4}\s*[-\s]\s*\d{6,8})",
    ]
    data["phone"] = find_first_match(phone_patterns, text)

    # ----------------------------
    # Invoice Number
    # ----------------------------
    invoice_patterns = [
        r"Cash\s*Bill\s*[:#]?\s*([A-Za-z0-9\-]+)",
        r"Invoice\s*No\.?\s*[:#]?\s*([A-Za-z0-9\-]+)",
        r"Bill\s*No\.?\s*[:#]?\s*([A-Za-z0-9\-]+)",
    ]
    data["invoice_number"] = find_first_match(invoice_patterns, text)

    # ----------------------------
    # Date & Time
    # ----------------------------
    date_match = re.search(r"(\d{2}[/-]\d{2}[/-]\d{4})", text)
    if date_match:
        data["invoice_date"] = date_match.group(1)

    time_match = re.search(
        r"(\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM)?)", text, re.IGNORECASE
    )
    if time_match:
        data["invoice_time"] = time_match.group(1)

    # ----------------------------
    # Cashier
    # ----------------------------
    cashier_patterns = [r"Cashier\s*[:#]?\s*([A-Za-z0-9]+)"]
    data["cashier"] = find_first_match(cashier_patterns, text)

    # ----------------------------
    # GST
    # ----------------------------
    gst_patterns = [
        r"GST\s*(?:ID|No)?\.?\s*[:#]?\s*([0-9A-Za-z]{10,15})",
        r"Reg\s*No\.?\s*[:#]?\s*([0-9A-Za-z\-]+)",
    ]
    data["gst_number"] = find_first_match(gst_patterns, text)

    # ----------------------------
    # Totals & Cash
    # ----------------------------
    total_patterns = [
        r"Total\s*Amount\s*[:#]?\s*([0-9]+\.[0-9]{2})",
        r"Total\s*[:#]?\s*([0-9]+\.[0-9]{2})",
    ]
    data["total_amount"] = find_first_match(total_patterns, text)

    cash_patterns = [
        r"Cash\s*Received\s*[:#]?\s*([0-9]+\.[0-9]{2})",
        r"Cash\s*[:#]?\s*([0-9]+\.[0-9]{2})",
    ]
    data["cash_received"] = find_first_match(cash_patterns, text)

    change_patterns = [r"Change\s*[:#]?\s*([0-9]+\.[0-9]{2})"]
    data["change"] = find_first_match(change_patterns, text)

    # ----------------------------
    # Items Parsing
    # ----------------------------
    # Matches: [Description] [Qty] [Price] [Amount]
    # Example: "Plastic   2   15.50   31.00"
    item_pattern = re.compile(
        r"^([A-Za-z0-9\s/&\-\.\(\)]+?)\s{2,}(\d+)\s+([0-9]+\.[0-9]{2})\s+([0-9]+\.[0-9]{2})$"
    )

    # Fallback pattern if Qty is missing: [Description] [Price/Amount]
    fallback_item_pattern = re.compile(
        r"^([A-Za-z0-9\s/&\-\.\(\)]+?)\s{2,}([0-9]+\.[0-9]{2})$"
    )

    for line in lines:
        # Ignore structural keyword lines
        if any(
            kw in line.lower()
            for kw in ["total", "subtotal", "cash", "change", "description"]
        ):
            continue

        match = item_pattern.search(line)
        if match:
            data["items"].append(
                {
                    "description": match.group(1).strip(),
                    "qty": match.group(2).strip(),
                    "price": match.group(3).strip(),
                    "amount": match.group(4).strip(),
                }
            )
        else:
            fallback_match = fallback_item_pattern.search(line)
            if fallback_match:
                data["items"].append(
                    {
                        "description": fallback_match.group(1).strip(),
                        "qty": "1",
                        "price": fallback_match.group(2).strip(),
                        "amount": fallback_match.group(2).strip(),
                    }
                )

    return data