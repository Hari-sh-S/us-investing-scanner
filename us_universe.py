# US Universe Data - Major US Stock Indices
# Last Updated: 2025-12-16
# Source: Major US Stock Indices - tickers use standard Yahoo Finance format (no suffix needed)

# All US indices
INDEX_NAMES = [
    "S&P 500",
    "NASDAQ 100",
    "DOW 30",
    "Russell 2000",
    "S&P MidCap 400",
    "S&P SmallCap 600",
    "S&P 500 Technology",
    "S&P 500 Healthcare",
    "S&P 500 Financials",
    "S&P 500 Consumer Discretionary",
    "S&P 500 Energy",
    "NASDAQ Financial",
    "NYSE FANG+",
]

# S&P 500 Top 50 (by market cap)
SP500_TOP_50 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
    "JPM", "V", "XOM", "MA", "PG", "HD", "CVX", "LLY", "MRK", "ABBV",
    "AVGO", "PEP", "KO", "COST", "WMT", "TMO", "MCD", "CSCO", "ACN", "CRM",
    "ABT", "DHR", "ADBE", "NFLX", "AMD", "INTC", "VZ", "WFC", "CMCSA", "TXN",
    "PM", "NEE", "BA", "RTX", "UNP", "ORCL", "HON", "IBM", "AMGN", "QCOM"
]

# NASDAQ 100 (Full list)
NASDAQ_100 = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "COST",
    "PEP", "NFLX", "AMD", "ADBE", "CSCO", "CMCSA", "TMUS", "INTC", "AMGN", "QCOM",
    "TXN", "HON", "INTU", "SBUX", "ISRG", "ADP", "BKNG", "VRTX", "AMAT", "GILD",
    "ADI", "MDLZ", "REGN", "LRCX", "MU", "CSX", "PYPL", "SNPS", "PANW", "KLAC",
    "CDNS", "MAR", "MNST", "ORLY", "KDP", "NXPI", "ADSK", "CTAS", "MELI", "AEP",
    "ASML", "FTNT", "PCAR", "CHTR", "KHC", "LULU", "MCHP", "PAYX", "EXC", "WDAY",
    "ROST", "AZN", "CPRT", "DXCM", "IDXX", "MRNA", "BIIB", "ODFL", "CRWD", "CTSH",
    "BKR", "FAST", "CSGP", "VRSK", "CEG", "TTD", "DDOG", "XEL", "TEAM", "FANG",
    "ON", "EA", "GEHC", "ZS", "ANSS", "GFS", "ILMN", "WBD", "EBAY", "ALGN",
    "ENPH", "CDW", "ZM", "DLTR", "SIRI", "JD", "LCID", "RIVN", "WBA", "MTCH"
]

# DOW 30
DOW_30 = [
    "AAPL", "MSFT", "AMZN", "V", "JPM", "JNJ", "UNH", "HD", "PG", "MRK",
    "CVX", "KO", "DIS", "CSCO", "VZ", "CRM", "MCD", "NKE", "BA", "IBM",
    "CAT", "GS", "MMM", "HON", "WMT", "AXP", "TRV", "INTC", "WBA", "DOW"
]

# Russell 2000 Sample (top 100 by liquidity as proxy)
RUSSELL_2000_SAMPLE = [
    "AMC", "RIOT", "MARA", "SOFI", "PLUG", "SNDL", "UPST", "CLOV", "BBBY", "GME",
    "SPCE", "BYND", "FUBO", "WKHS", "SKLZ", "CLNE", "VXRT", "GSAT", "BLNK", "GOEV",
    "NKLA", "RIDE", "OPEN", "BFLY", "VUZI", "SRNE", "OCGN", "TTCF", "BNGO", "PRTY",
    "LGVN", "CRBP", "CPRX", "CODX", "EVFM", "FCEL", "MNKD", "CRSP", "NTLA", "EDIT",
    "BEAM", "FATE", "BLUE", "SGEN", "ALNY", "EXAS", "IONS", "ARWR", "SRPT", "BMRN",
    "SMAR", "OKTA", "NEWR", "PATH", "SNOW", "BILL", "MDB", "NET", "FSLY", "ESTC",
    "DKNG", "PENN", "RSI", "GENI", "CHGG", "MTCH", "IAC", "BMBL", "UBER", "LYFT",
    "DASH", "ABNB", "COIN", "RBLX", "U", "DOCS", "DOCN", "CFLT", "GTLB", "APPS",
    "MGNI", "PUBM", "TTD", "ROKU", "SPOT", "SQ", "AFRM", "HOOD", "PAYO", "LSPD",
    "BIGC", "VTEX", "SHOP", "ETSY", "WISH", "POSH", "REAL", "CVNA", "CARG", "VRM"
]

# S&P MidCap 400 Sample (top 50)
SP_MIDCAP_400_SAMPLE = [
    "POOL", "TRMB", "WSM", "DPZ", "RRC", "EQT", "AR", "WYNN", "MGM", "LVS",
    "CZR", "RRX", "INGR", "POST", "LNTH", "ZWS", "ELF", "SHAK", "CAVA", "CMC",
    "STLD", "RS", "WWD", "LECO", "AGCO", "HRI", "ALK", "JBLU", "LUV", "HA",
    "DINO", "PBF", "DK", "HFC", "CIVI", "PDCE", "OVV", "MTDR", "FANG", "SM",
    "CEIX", "ARCH", "BTU", "ARLP", "SXC", "HCC", "CNX", "NFE", "KOS", "WLL"
]

# S&P SmallCap 600 Sample (top 50)
SP_SMALLCAP_600_SAMPLE = [
    "XPEL", "SHOO", "HIBB", "DDS", "BGFV", "SCVL", "CAL", "CROX", "SKX", "DECK",
    "BOOT", "MOV", "FOSL", "SGC", "SNBR", "LQDT", "PETS", "PRTS", "GIII", "OXM",
    "GCO", "JILL", "ZUMZ", "URBN", "EXPR", "XPOF", "FRGI", "CAKE", "TXRH", "EAT",
    "DRI", "BLMN", "DIN", "ARCO", "PHTF", "LOCO", "JACK", "RRGB", "SBUX", "DENN",
    "NDLS", "FRSH", "BROS", "DAVE", "PTLO", "SHAK", "WING", "CAVA", "KRUS", "TAST"
]

# Technology Sector (S&P 500 Tech)
SP500_TECH = [
    "AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "CRM", "AMD", "INTC", "CSCO", "TXN",
    "QCOM", "ORCL", "IBM", "NOW", "INTU", "AMAT", "MU", "ADI", "LRCX", "KLAC",
    "SNPS", "CDNS", "APH", "MSI", "FTNT", "PANW", "MCHP", "GLW", "KEYS", "HPQ",
    "HPE", "WDC", "STX", "NTAP", "ZBRA", "JNPR", "CDW", "AKAM", "FFIV", "VRSN"
]

# Healthcare Sector (S&P 500 Healthcare)
SP500_HEALTHCARE = [
    "LLY", "UNH", "JNJ", "MRK", "ABBV", "PFE", "TMO", "DHR", "ABT", "BMY",
    "AMGN", "MDT", "ISRG", "GILD", "VRTX", "REGN", "CVS", "CI", "ELV", "SYK",
    "ZTS", "BDX", "BSX", "HCA", "MCK", "EW", "DXCM", "A", "IQV", "IDXX",
    "MTD", "WAT", "CAH", "ABC", "HUM", "COR", "RMD", "ALGN", "WST", "HOLX"
]

# Financials Sector (S&P 500 Financials)
SP500_FINANCIALS = [
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "SPGI", "AXP",
    "BLK", "C", "SCHW", "CB", "MMC", "PGR", "USB", "CME", "AON", "PNC",
    "ICE", "MET", "AIG", "TRV", "AJG", "COF", "AFL", "PRU", "ALL", "MSCI",
    "BK", "TFC", "FI", "MCO", "WTW", "STT", "FIS", "CINF", "NDAQ", "RJF"
]

# Consumer Discretionary Sector
SP500_CONSUMER_DISC = [
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "MAR",
    "CMG", "ORLY", "AZO", "DHI", "LEN", "F", "GM", "ROST", "YUM", "HLT",
    "DG", "DLTR", "EBAY", "ULTA", "POOL", "BBY", "APTV", "GPC", "LKQ", "GRMN",
    "PHM", "NVR", "TPR", "RL", "VFC", "PVH", "HAS", "WYNN", "LVS", "MGM"
]

# Energy Sector
SP500_ENERGY = [
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PXD", "VLO", "PSX", "OXY",
    "HES", "WMB", "KMI", "DVN", "HAL", "FANG", "BKR", "CTRA", "MRO", "OKE",
    "TRGP", "APA", "EQT", "PR", "OVV"
]

# FANG+ Stocks (NYSE FANG+ Index)
FANG_PLUS = [
    "META", "AMZN", "NFLX", "GOOGL", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AVGO"
]

# Universe dictionary
UNIVERSES = {name: [] for name in INDEX_NAMES}

# Populate universes
UNIVERSES["S&P 500"] = SP500_TOP_50  # Using top 50 for now
UNIVERSES["NASDAQ 100"] = NASDAQ_100
UNIVERSES["DOW 30"] = DOW_30
UNIVERSES["Russell 2000"] = RUSSELL_2000_SAMPLE
UNIVERSES["S&P MidCap 400"] = SP_MIDCAP_400_SAMPLE
UNIVERSES["S&P SmallCap 600"] = SP_SMALLCAP_600_SAMPLE
UNIVERSES["S&P 500 Technology"] = SP500_TECH
UNIVERSES["S&P 500 Healthcare"] = SP500_HEALTHCARE
UNIVERSES["S&P 500 Financials"] = SP500_FINANCIALS
UNIVERSES["S&P 500 Consumer Discretionary"] = SP500_CONSUMER_DISC
UNIVERSES["S&P 500 Energy"] = SP500_ENERGY
UNIVERSES["NASDAQ Financial"] = ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "BK", "SCHW"]
UNIVERSES["NYSE FANG+"] = FANG_PLUS


def get_universe(name):
    """Get stock list for a given universe name."""
    return UNIVERSES.get(name, [])


def get_all_universe_names():
    """Get all available universe names in the specified order."""
    return INDEX_NAMES.copy()


def get_broad_market_universes():
    """Get broad market indices."""
    return [
        "S&P 500", "NASDAQ 100", "DOW 30", "Russell 2000",
        "S&P MidCap 400", "S&P SmallCap 600"
    ]


def get_sectoral_universes():
    """Get sectoral/thematic indices."""
    return [
        "S&P 500 Technology", "S&P 500 Healthcare", "S&P 500 Financials",
        "S&P 500 Consumer Discretionary", "S&P 500 Energy"
    ]


def get_cap_based_universes():
    """Get cap-based indices."""
    return [
        "S&P 500", "S&P MidCap 400", "S&P SmallCap 600", "Russell 2000"
    ]


def get_thematic_universes():
    """Get strategy/thematic indices."""
    return [
        "NYSE FANG+", "NASDAQ Financial"
    ]
