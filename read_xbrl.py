"""Parse UK nonprofit/charity iXBRL filings for nonprofit explorer data.

Extracts organizational identity, officers/trustees, financial summary,
and filing metadata from UK Companies House iXBRL account filings.
Analogous to data available through ProPublica's Nonprofit Explorer
for US nonprofits, adapted for UK Companies House and Charity Commission filings.
"""

from __future__ import annotations

import datetime
import decimal
import io
import logging
import re
import typing
from dataclasses import dataclass

import dateutil.parser
import lxml.etree

if typing.TYPE_CHECKING:
    from lxml.etree import _Element as Element

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NonprofitFiling:
    """Parsed data from a UK nonprofit/charity iXBRL filing.

    Fields are grouped by category: identity, period, status, people,
    address, current-period financials, prior-period comparatives,
    audit/signing dates, and metadata.
    """

    # Identity
    companies_house_number: str | None
    charity_number: str | None
    entity_name: str | None
    legal_form: str | None

    # Period
    period_start: datetime.date | None
    period_end: datetime.date | None
    balance_sheet_date: datetime.date | None

    # Status
    is_dormant: bool | None
    is_audited: bool | None

    # People
    officers: tuple[str, ...]
    auditor_name: str | None
    senior_statutory_auditor: str | None
    examiner_name: str | None
    accountant_name: str | None

    # Address
    registered_address: str | None

    # Financial - Current Period
    charity_funds: decimal.Decimal | None
    net_assets: decimal.Decimal | None
    equity: decimal.Decimal | None
    fixed_assets: decimal.Decimal | None
    current_assets: decimal.Decimal | None
    cash_bank_in_hand: decimal.Decimal | None
    debtors: decimal.Decimal | None
    trade_debtors: decimal.Decimal | None
    creditors_due_within_one_year: decimal.Decimal | None
    trade_creditors: decimal.Decimal | None
    net_current_assets: decimal.Decimal | None
    total_assets_less_current_liabilities: decimal.Decimal | None
    average_employees: decimal.Decimal | None

    # Financial - Prior Period (key comparatives)
    charity_funds_prior: decimal.Decimal | None
    net_assets_prior: decimal.Decimal | None
    equity_prior: decimal.Decimal | None
    current_assets_prior: decimal.Decimal | None
    average_employees_prior: decimal.Decimal | None

    # Audit / Signing
    accounts_signing_date: datetime.date | None
    auditors_report_date: datetime.date | None

    # Metadata
    production_software: str | None


# Tag-to-field mappings, keyed by local name (after stripping namespace prefix).
# Each handles multiple namespace conventions (bus:, uk-bus:, ns10:, core:, uk-core:, ns5:, char:).

_STR_TAG_MAP: dict[str, str] = {
    "UKCompaniesHouseRegisteredNumber": "companies_house_number",
    "CompaniesHouseRegisteredNumber": "companies_house_number",
    "CharityRegistrationNumberEnglandWales": "charity_number",
    "EntityCurrentLegalOrRegisteredName": "entity_name",
    "EntityCurrentLegalName": "entity_name",
    "LegalFormEntity": "legal_form",
    "NameEntityAuditors": "auditor_name",
    "NameSeniorStatutoryAuditor": "senior_statutory_auditor",
    "NameEntityExaminers": "examiner_name",
    "NameEntityAccountants": "accountant_name",
    "NameProductionSoftware": "production_software",
}

_DATE_TAG_MAP: dict[str, str] = {
    "StartDateForPeriodCoveredByReport": "period_start",
    "EndDateForPeriodCoveredByReport": "period_end",
    "BalanceSheetDate": "balance_sheet_date",
    "DateAuthorisationFinancialStatementsForIssue": "accounts_signing_date",
    "DateAuditorsReport": "auditors_report_date",
}

_BOOL_TAG_MAP: dict[str, str] = {
    "EntityDormantTruefalse": "is_dormant",
    "EntityDormant": "is_dormant",
}

_FINANCIAL_TAG_MAP: dict[str, str] = {
    "CharityFunds": "charity_funds",
    "NetAssetsLiabilities": "net_assets",
    "NetAssetsLiabilitiesIncludingPensionAssetLiability": "net_assets",
    "Equity": "equity",
    "ShareholderFunds": "equity",
    "FixedAssets": "fixed_assets",
    "TangibleFixedAssets": "fixed_assets",
    "PropertyPlantEquipment": "fixed_assets",
    "CurrentAssets": "current_assets",
    "CashBankInHand": "cash_bank_in_hand",
    "CashBankOnHand": "cash_bank_in_hand",
    "Debtors": "debtors",
    "TradeDebtorsTradeReceivables": "trade_debtors",
    "CreditorsDueWithinOneYear": "creditors_due_within_one_year",
    "Creditors": "creditors_due_within_one_year",
    "TradeCreditorsTradePayables": "trade_creditors",
    "NetCurrentAssetsLiabilities": "net_current_assets",
    "TotalAssetsLessCurrentLiabilities": "total_assets_less_current_liabilities",
    "AverageNumberEmployeesDuringPeriod": "average_employees",
    "EmployeesTotal": "average_employees",
}

_ADDRESS_TAG_MAP: dict[str, str] = {
    "AddressLine1": "line1",
    "AddressLine2": "line2",
    "AddressLine3": "line3",
    "PrincipalLocation-CityOrTown": "city",
    "CountyRegion": "county",
    "PostalCodeZip": "postal_code",
}

_PRIOR_FINANCIAL_FIELDS = frozenset({
    "charity_funds",
    "net_assets",
    "equity",
    "current_assets",
    "average_employees",
})


def _get_text(element: Element) -> str:
    """Extract text content from an element, skipping ix:exclude children."""
    parts: list[str] = []
    for e in element.iter():
        if not isinstance(e.tag, str):
            continue
        local = e.tag.rpartition("}")[2] or e.tag
        if local == "exclude":
            continue
        if e.text:
            parts.append(e.text)
    return "".join(parts).strip().replace("\n", " ").replace('"', "")


def _parse_decimal_value(element: Element, text: str) -> decimal.Decimal | None:
    """Parse a decimal value from an iXBRL numeric element.

    Handles sign, scale, and format attributes per the iXBRL spec.
    """
    text = text.strip()
    if not text or text in ("-", "\u2014", ""):
        return None

    sign = -1 if element.get("sign", "") == "-" else 1
    fmt = element.get("format", "").rpartition(":")[2]

    if fmt == "numdotcomma":
        text = text.replace(".", "").replace(",", ".")
    elif fmt == "numspacedot":
        text = text.replace(" ", "")
    else:
        text = text.replace(",", "")

    try:
        value = decimal.Decimal(text)
    except decimal.InvalidOperation:
        return None

    scale = element.get("scale", "0")
    try:
        return sign * value * decimal.Decimal(10) ** decimal.Decimal(scale)
    except (decimal.InvalidOperation, decimal.Overflow):
        return None


def _parse_date_text(text: str) -> datetime.date | None:
    """Parse a date from various text formats used in UK iXBRL filings."""
    text = text.strip()
    if not text:
        return None

    try:
        return datetime.date.fromisoformat(text)
    except ValueError:
        pass

    text = re.sub(r"(?i)(\d)((st)|(nd)|(rd)|(th))", r"\1", text)
    try:
        return dateutil.parser.parse(text, dayfirst=True).date()
    except (ValueError, dateutil.parser.ParserError):
        return None


def _build_context_end_dates(document: lxml.etree._ElementTree) -> dict[str, datetime.date]:
    """Map context IDs to their period end dates (or instant dates)."""
    context_end_dates: dict[str, datetime.date] = {}

    for ctx in typing.cast(
        "list[Element]",
        document.xpath("//*[local-name()='context']"),
    ):
        ctx_id = ctx.get("id")
        if not ctx_id:
            continue
        for period in typing.cast("list[Element]", ctx.xpath("./*[local-name()='period']")):
            instants = typing.cast("list[str]", period.xpath("./*[local-name()='instant']/text()"))
            if instants:
                try:
                    context_end_dates[ctx_id] = datetime.date.fromisoformat(instants[0].strip())
                except ValueError:
                    pass
                continue
            ends = typing.cast("list[str]", period.xpath("./*[local-name()='endDate']/text()"))
            if ends:
                try:
                    context_end_dates[ctx_id] = datetime.date.fromisoformat(ends[0].strip())
                except ValueError:
                    pass
    return context_end_dates


def parse_filing(xbrl_bytes: bytes) -> NonprofitFiling:
    """Parse an iXBRL/XBRL file and extract nonprofit-relevant data.

    Works with any UK Companies House iXBRL filing regardless of filename
    convention or namespace prefix style (bus:, uk-bus:, ns10:, etc.).

    Args:
        xbrl_bytes: Raw bytes of the iXBRL/XBRL file.

    Returns:
        A NonprofitFiling with all extractable data populated.
    """
    start = xbrl_bytes.find(b"<")
    if start < 0:
        return _empty_filing()

    try:
        document = lxml.etree.parse(
            io.BytesIO(xbrl_bytes[start:]),
            lxml.etree.XMLParser(ns_clean=True, recover=True),
        )
        document.xpath("//*[0]")
    except (lxml.etree.Error, AssertionError):
        logger.warning("Failed to parse XBRL document")
        return _empty_filing()

    context_end_dates = _build_context_end_dates(document)

    unique_end_dates = sorted(set(context_end_dates.values()), reverse=True)
    current_end = unique_end_dates[0] if unique_end_dates else None
    prior_end = unique_end_dates[1] if len(unique_end_dates) > 1 else None

    str_fields: dict[str, str] = {}
    date_fields: dict[str, datetime.date] = {}
    bool_fields: dict[str, bool] = {}
    officers: list[str] = []
    address_parts: dict[str, str] = {}
    financial_current: dict[str, decimal.Decimal] = {}
    financial_prior: dict[str, decimal.Decimal] = {}
    has_audit_exemption = False
    has_auditors_report = False

    for element in typing.cast("list[Element]", document.xpath("//*")):
        tag_str = element.tag
        if not isinstance(tag_str, str):
            continue

        tag_local = tag_str.rpartition("}")[2] or tag_str
        name_attr = element.get("name", "")
        attr_local = name_attr.rpartition(":")[2] if name_attr else ""
        ctx_ref = element.get("contextRef", "")

        for local_name in (tag_local, attr_local):
            if not local_name:
                continue

            if local_name == "NameEntityOfficer":
                text = _get_text(element)
                if text and text not in officers:
                    officers.append(text)
                continue

            if local_name.startswith("StatementThatCompanyEntitledToExemptionFromAudit"):
                has_audit_exemption = True
                continue

            if local_name == "DateAuditorsReport":
                has_auditors_report = True

            if local_name in _STR_TAG_MAP:
                field = _STR_TAG_MAP[local_name]
                if field not in str_fields:
                    text = _get_text(element)
                    if text:
                        str_fields[field] = text
                continue

            if local_name in _DATE_TAG_MAP:
                field = _DATE_TAG_MAP[local_name]
                if field not in date_fields:
                    text = _get_text(element)
                    parsed = _parse_date_text(text) if text else None
                    if parsed is not None:
                        date_fields[field] = parsed
                continue

            if local_name in _BOOL_TAG_MAP:
                field = _BOOL_TAG_MAP[local_name]
                if field not in bool_fields:
                    text = _get_text(element).lower()
                    if text in ("true", "false"):
                        bool_fields[field] = text == "true"
                continue

            if local_name in _ADDRESS_TAG_MAP:
                slot = _ADDRESS_TAG_MAP[local_name]
                if slot not in address_parts:
                    text = _get_text(element)
                    if text:
                        address_parts[slot] = text
                continue

            if local_name in _FINANCIAL_TAG_MAP:
                field = _FINANCIAL_TAG_MAP[local_name]
                text = _get_text(element)
                value = _parse_decimal_value(element, text) if text else None
                if value is not None and ctx_ref:
                    end_date = context_end_dates.get(ctx_ref)
                    if end_date == current_end and field not in financial_current:
                        financial_current[field] = value
                    elif end_date == prior_end and field not in financial_prior:
                        financial_prior[field] = value
                continue

    address_line_parts = [
        address_parts.get(k, "")
        for k in ("line1", "line2", "line3", "city", "county", "postal_code")
    ]
    registered_address = ", ".join(p for p in address_line_parts if p) or None

    is_audited: bool | None = None
    if has_auditors_report:
        is_audited = True
    elif has_audit_exemption:
        is_audited = False

    return NonprofitFiling(
        companies_house_number=str_fields.get("companies_house_number"),
        charity_number=str_fields.get("charity_number"),
        entity_name=str_fields.get("entity_name"),
        legal_form=str_fields.get("legal_form"),
        period_start=date_fields.get("period_start"),
        period_end=date_fields.get("period_end"),
        balance_sheet_date=date_fields.get("balance_sheet_date"),
        is_dormant=bool_fields.get("is_dormant"),
        is_audited=is_audited,
        officers=tuple(officers),
        auditor_name=str_fields.get("auditor_name"),
        senior_statutory_auditor=str_fields.get("senior_statutory_auditor"),
        examiner_name=str_fields.get("examiner_name"),
        accountant_name=str_fields.get("accountant_name"),
        registered_address=registered_address,
        charity_funds=financial_current.get("charity_funds"),
        net_assets=financial_current.get("net_assets"),
        equity=financial_current.get("equity"),
        fixed_assets=financial_current.get("fixed_assets"),
        current_assets=financial_current.get("current_assets"),
        cash_bank_in_hand=financial_current.get("cash_bank_in_hand"),
        debtors=financial_current.get("debtors"),
        trade_debtors=financial_current.get("trade_debtors"),
        creditors_due_within_one_year=financial_current.get("creditors_due_within_one_year"),
        trade_creditors=financial_current.get("trade_creditors"),
        net_current_assets=financial_current.get("net_current_assets"),
        total_assets_less_current_liabilities=financial_current.get("total_assets_less_current_liabilities"),
        average_employees=financial_current.get("average_employees"),
        charity_funds_prior=financial_prior.get("charity_funds"),
        net_assets_prior=financial_prior.get("net_assets"),
        equity_prior=financial_prior.get("equity"),
        current_assets_prior=financial_prior.get("current_assets"),
        average_employees_prior=financial_prior.get("average_employees"),
        accounts_signing_date=date_fields.get("accounts_signing_date"),
        auditors_report_date=date_fields.get("auditors_report_date"),
        production_software=str_fields.get("production_software"),
    )


def _empty_filing() -> NonprofitFiling:
    """Return an empty NonprofitFiling with all fields set to None."""
    return NonprofitFiling(
        companies_house_number=None,
        charity_number=None,
        entity_name=None,
        legal_form=None,
        period_start=None,
        period_end=None,
        balance_sheet_date=None,
        is_dormant=None,
        is_audited=None,
        officers=(),
        auditor_name=None,
        senior_statutory_auditor=None,
        examiner_name=None,
        accountant_name=None,
        registered_address=None,
        charity_funds=None,
        net_assets=None,
        equity=None,
        fixed_assets=None,
        current_assets=None,
        cash_bank_in_hand=None,
        debtors=None,
        trade_debtors=None,
        creditors_due_within_one_year=None,
        trade_creditors=None,
        net_current_assets=None,
        total_assets_less_current_liabilities=None,
        average_employees=None,
        charity_funds_prior=None,
        net_assets_prior=None,
        equity_prior=None,
        current_assets_prior=None,
        average_employees_prior=None,
        accounts_signing_date=None,
        auditors_report_date=None,
        production_software=None,
    )


if __name__ == "__main__":
    import pathlib
    import sys

    files = sys.argv[1:] if len(sys.argv) > 1 else []
    for filepath in files:
        data = pathlib.Path(filepath).read_bytes()
        filing = parse_filing(data)
        print(f"\n{'=' * 70}")
        print(f"File: {filepath}")
        print(f"{'=' * 70}")
        print(f"  Entity:          {filing.entity_name}")
        print(f"  CH Number:       {filing.companies_house_number}")
        print(f"  Charity Number:  {filing.charity_number}")
        print(f"  Legal Form:      {filing.legal_form}")
        print(f"  Period:          {filing.period_start} to {filing.period_end}")
        print(f"  Balance Sheet:   {filing.balance_sheet_date}")
        print(f"  Dormant:         {filing.is_dormant}")
        print(f"  Audited:         {filing.is_audited}")
        print(f"  Officers:        {', '.join(filing.officers) if filing.officers else 'None'}")
        print(f"  Auditor:         {filing.auditor_name}")
        print(f"  Stat. Auditor:   {filing.senior_statutory_auditor}")
        print(f"  Examiner:        {filing.examiner_name}")
        print(f"  Accountant:      {filing.accountant_name}")
        print(f"  Address:         {filing.registered_address}")
        print(f"  --- Current Period ---")
        print(f"  Charity Funds:   {filing.charity_funds}")
        print(f"  Net Assets:      {filing.net_assets}")
        print(f"  Equity:          {filing.equity}")
        print(f"  Fixed Assets:    {filing.fixed_assets}")
        print(f"  Current Assets:  {filing.current_assets}")
        print(f"  Cash:            {filing.cash_bank_in_hand}")
        print(f"  Debtors:         {filing.debtors}")
        print(f"  Trade Debtors:   {filing.trade_debtors}")
        print(f"  Creditors (<1y): {filing.creditors_due_within_one_year}")
        print(f"  Trade Creditors: {filing.trade_creditors}")
        print(f"  Net Current:     {filing.net_current_assets}")
        print(f"  Assets-Curr Liab:{filing.total_assets_less_current_liabilities}")
        print(f"  Avg Employees:   {filing.average_employees}")
        print(f"  --- Prior Period ---")
        print(f"  Charity Funds:   {filing.charity_funds_prior}")
        print(f"  Net Assets:      {filing.net_assets_prior}")
        print(f"  Equity:          {filing.equity_prior}")
        print(f"  Current Assets:  {filing.current_assets_prior}")
        print(f"  Avg Employees:   {filing.average_employees_prior}")
        print(f"  --- Dates ---")
        print(f"  Signing Date:    {filing.accounts_signing_date}")
        print(f"  Auditors Report: {filing.auditors_report_date}")
        print(f"  Software:        {filing.production_software}")
