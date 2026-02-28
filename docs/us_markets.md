# US Market Support — DataPAI

DataPAI supports both **ASX (Australian Securities Exchange)** and **US stock exchanges (NYSE / NASDAQ)** via two parallel pipelines that share the same technical analysis engine and LLM infrastructure.

---

## Architecture Overview

```
                    ┌──────────────────────────────┐
                    │   Market Announcements Tab    │
                    └──────────────┬───────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                                          │
    ┌─────────▼─────────┐                   ┌──────────▼──────────┐
    │  ASX Pipeline      │                   │  US Pipeline         │
    │ (asx_announcement_ │                   │ (sec_filing_agent.py)│
    │  agent.py)         │                   │                      │
    │                    │                   │ Source: SEC EDGAR    │
    │ Source: ASX.com.au │                   │ Forms: 8-K, 10-Q,    │
    │ Format: PDF        │                   │        10-K, DEF 14A │
    │ Currency: AUD      │                   │ Format: HTML/TXT     │
    └─────────┬──────────┘                   │ Currency: USD        │
              │                              └──────────┬───────────┘
              │                                         │
              └────────────────┬────────────────────────┘
                               │
               ┌───────────────▼────────────────┐
               │   agents/technical_analysis.py  │
               │                                 │
               │  fetch_all_timeframes()          │
               │  calc_indicators()               │
               │  RSI · MACD · BB · EMAs          │
               │  generate_technical_signal()     │
               └───────────────┬─────────────────┘
                               │
               ┌───────────────▼─────────────────┐
               │   agents/data_providers/         │
               │                                 │
               │  yahoo.py   — yfinance (free)    │
               │  polygon.py — Polygon.io (paid)  │
               │  eodhd.py   — EODHD (paid, ASX) │
               └───────────────┬─────────────────┘
                               │
               ┌───────────────▼─────────────────┐
               │   Gemini + GPT LLM chain         │
               │                                 │
               │  Gemini: interpret + ground      │
               │  (Google Search for live news)   │
               │  GPT: compliance reviewer        │
               └─────────────────────────────────┘
```

---

## US Market — SEC EDGAR Filing Agent

### What is SEC EDGAR?

The SEC (Securities and Exchange Commission) requires all US-listed public companies to file disclosures electronically via EDGAR (Electronic Data Gathering, Analysis, and Retrieval system). EDGAR is **free, public, and has a REST API**.

This is the US equivalent of ASX market announcements — when Apple reports earnings, files an 8-K that discloses a major acquisition, or changes its CEO, that filing appears on EDGAR within minutes.

### Form Types

| Form | Name | Frequency | Market Sensitivity |
|------|------|-----------|-------------------|
| **8-K** | Current report | As events occur | ⭐ Highest — material events in real time |
| **10-Q** | Quarterly report | 4× per year | High — audited financials |
| **10-K** | Annual report | 1× per year | High — comprehensive audited financials |
| **DEF 14A** | Proxy statement | 1× per year | Medium — shareholder votes, CEO pay |
| **S-1** | IPO registration | One-time | Very high — IPO pricing |

### 8-K Item Numbers

8-K filings include item numbers indicating what happened. DataPAI maps these to human-readable headlines and uses them to classify market sensitivity:

| Item | Description | Market Sensitive |
|------|-------------|-----------------|
| 1.01 | Entry into Material Agreement (M&A, JV, major contract) | ✅ Yes |
| 1.02 | Termination of Material Agreement | ✅ Yes |
| 2.01 | Completion of Acquisition or Disposition | ✅ Yes |
| **2.02** | **Results of Operations — EARNINGS** | ✅ Yes |
| 2.03 | Creation of Direct Financial Obligation (new debt) | ✅ Yes |
| 2.06 | Material Impairments / Write-downs | ✅ Yes |
| 4.01 | Change in Certifying Accountant | ✅ Yes |
| **4.02** | **Non-Reliance on Financial Statements — RESTATEMENT RISK** | ✅ Yes ⚠️ |
| 5.01 | Change in Control / Takeover | ✅ Yes |
| 5.02 | Management Change (CEO/CFO departure or election) | ⚠️ Maybe |
| 7.01 | Regulation FD Disclosure | Maybe |
| 8.01 | Other Events | Maybe |
| 9.01 | Financial Statements and Exhibits (boilerplate) | ❌ No |

### API

SEC EDGAR REST API is **free, no API key required**, but SEC policy requires a descriptive `User-Agent` header. Set this env var:

```bash
export SEC_CONTACT_EMAIL="yourname@yourcompany.com"
```

If not set, `datapai@example.com` is used as a fallback (works but not polite to SEC).

API endpoints used:

```
# Bulk ticker → CIK mapping (loaded once per session, ~2 MB)
https://www.sec.gov/files/company_tickers.json

# Company submissions (filings list)
https://data.sec.gov/submissions/CIK{cik:010d}.json

# Filing document
https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{primary_doc}
```

---

## Data Sources for US Stocks

### Yahoo Finance (default, free)

```python
fetch_all_timeframes("AAPL", suffix="", source="yahoo")
```

- **No API key needed**
- Works for NYSE, NASDAQ, AMEX
- Intraday: 60-day limit for 5m/30m bars
- Best free option for US stocks

### Polygon.io (paid, US only)

```bash
export POLYGON_API_KEY="your_key"
```

```python
fetch_all_timeframes("AAPL", suffix="", source="polygon")
```

- Real-time data (paid tier) or 15-min delayed (free tier)
- US exchanges only — **not suitable for ASX**
- Better intraday depth than Yahoo for US stocks
- Get a key at [polygon.io](https://polygon.io)

### EOD Historical Data (paid, best for ASX)

```bash
export EODHD_API_KEY="your_key"
```

```python
# ASX — EODHD uses ".AU" internally (mapped automatically from ".AX")
fetch_all_timeframes("BHP", suffix=".AX", source="eodhd")

# US
fetch_all_timeframes("AAPL", suffix="", source="eodhd")
```

- Better ASX intraday history than Yahoo (not capped at 60 days)
- 70+ exchanges worldwide
- Get a key at [eodhd.com](https://eodhd.com)

### Graceful fallback

If a paid provider's API key is not set, DataPAI automatically falls back to Yahoo Finance with a log warning — no code change needed:

```
WARNING: Data provider 'polygon' requires POLYGON_API_KEY which is not set.
         Falling back to Yahoo Finance.
```

---

## Exchange Suffixes Reference

| Exchange | Suffix | Example | Provider recommendation |
|----------|--------|---------|------------------------|
| ASX (Australia) | `.AX` | `BHP.AX` | Yahoo (free) or EODHD (paid) |
| NYSE / NASDAQ | *(none)* | `AAPL` | Yahoo or Polygon |
| London (LSE) | `.L` | `BP.L` | Yahoo |
| Toronto (TSX) | `.TO` | `TD.TO` | Yahoo |
| Hong Kong | `.HK` | `0700.HK` | Yahoo |
| New Zealand | `.NZ` | `AIR.NZ` | Yahoo or EODHD |
| Singapore | `.SI` | `D05.SI` | Yahoo or EODHD |

---

## Gemini Google Search Grounding

When `use_grounding=True` (default), Gemini searches Google in real-time to supplement our computed indicators with:

- Recent earnings beats/misses vs analyst consensus
- Latest analyst price target changes
- Breaking news (M&A rumours, regulatory actions, product launches)
- Sector and macro context

**Why this matters for US stocks specifically:** US markets have much higher analyst coverage and faster-moving news cycles than ASX. Grounding is particularly valuable for:

- Earnings reaction signals (did the stock beat/miss? what's the whisper number?)
- M&A speculation (is there deal premium baked in?)
- Fed/macro sensitivity (rate-sensitive sectors)
- Options market activity ahead of announcements

The signal always clearly marks which content came from grounding sources vs computed indicators.

---

## Technical Analysis (Tab 9)

Tab 9 already supports US stocks. Just:

1. Enter the ticker (e.g. `AAPL`, `NVDA`, `TSLA`)
2. Select **NYSE / NASDAQ** from the exchange dropdown (or leave blank suffix)
3. Click **Analyse**

The indicator math (RSI, MACD, Bollinger Bands, EMAs) is identical — fully exchange-agnostic pure pandas, no ta-lib required.

---

## Comparing ASX vs US Pipeline

| Feature | ASX | US (SEC EDGAR) |
|---------|-----|----------------|
| Filing source | ASX announcements page (HTML scrape) | SEC EDGAR REST API |
| File format | PDF | HTML / TXT (XBRL) |
| Announcement types | Market announcements, results, notices | 8-K, 10-Q, 10-K, proxy, S-1 |
| Frequency | As announced | As filed with SEC |
| API cost | Free | Free |
| Coverage | ASX-listed companies | All SEC-registered US public companies |
| Currency | AUD | USD |
| Intraday data (free) | Yahoo Finance (60-day limit) | Yahoo Finance (60-day limit) |
| Intraday data (paid) | EODHD (recommended) | Polygon.io (recommended) |
| LLM chain | Gemini (grounded) → GPT reviewer | Gemini (grounded) → GPT reviewer |

---

## Environment Variables

```bash
# LLM providers
export GOOGLE_API_KEY="..."        # Gemini (required for primary signal)
export GOOGLE_MODEL="gemini-2.0-flash"  # Supports grounding (recommended for signals)
export OPENAI_API_KEY="..."        # GPT reviewer (required)
export OPENAI_MODEL="gpt-4.1"

# Data providers (all optional — falls back to Yahoo Finance)
export POLYGON_API_KEY="..."       # Polygon.io (US stocks, paid)
export EODHD_API_KEY="..."         # EODHD (ASX + global, paid)

# SEC EDGAR (US announcements)
export SEC_CONTACT_EMAIL="you@company.com"   # Required by SEC policy

# Cost control
export DAILY_LLM_BUDGET_USD="5.00"
export COST_GUARD_ENABLED="true"
```

---

## Programmatic Usage

```python
# === US Filing Agent ===
from agents.sec_filing_agent import (
    get_cik,
    fetch_sec_filings,
    download_filing_text,
    interpret_filing,
    generate_us_trading_signal,
)

# Look up CIK
cik = get_cik("NVDA")          # → 1045810

# Fetch recent 8-K filings
filings = fetch_sec_filings("NVDA", count=10, form_types=("8-K",))

# Download and interpret
filing = filings[0]
text   = download_filing_text(filing["cik"], filing["accession"], filing["primary_doc"])
interp = interpret_filing(filing, text, use_grounding=True)

# === Technical Analysis (US) ===
from agents.technical_analysis import fetch_all_timeframes, generate_technical_signal

# US stock — empty suffix for NYSE/NASDAQ
indicators = fetch_all_timeframes("NVDA", suffix="", source="yahoo")
signal     = generate_technical_signal("NVDA", suffix="", use_grounding=True)

# Combined: filing + technicals
us_signal = generate_us_trading_signal(
    filing,
    text,
    indicators_by_tf=indicators,
    ticker_suffix="",
    use_grounding=True,
)
```

---

## Known Limitations

1. **SEC EDGAR intraday delays**: EDGAR filings appear with a short delay after submission (usually minutes). Real-time is faster via paid data vendors (not implemented yet).

2. **8-K exhibit parsing**: Some 8-K filings link to separate exhibit files (press releases, earnings tables) in item 9.01. DataPAI downloads the primary document; the press release exhibit (often the earnings table) may require an additional download. Future improvement.

3. **XBRL structured data**: 10-Q/10-K filings use XBRL (structured accounting data). DataPAI currently strips XBRL tags and processes plain text. Future improvement: parse XBRL for machine-readable financials.

4. **Non-US tickers in EDGAR**: Some ADRs and foreign private issuers (e.g. BHP's ADR `BHP` on NYSE) file with SEC. CIK lookup works for these, but technical data for the ADR (`BHP` on NYSE, suffix `""`) and the ASX-listed stock (`BHP.AX`) differ due to currency and share structure.

5. **Polygon.io free tier**: 15-minute delayed data. Use the paid tier for real-time signals.
