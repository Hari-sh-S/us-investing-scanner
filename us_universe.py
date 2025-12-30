# US Universe Data - Major US Stock Indices
# Last Updated: 2025-12-16
# Source: Major US Stock Indices - tickers use standard Yahoo Finance format (no suffix needed)

# All US indices
INDEX_NAMES = [
    # Market Cap Based
    "Mega Cap (>$200B)",
    "Large Cap ($10B-$200B)",
    "Medium Cap ($2B-$10B)",
    "Small Cap ($300M-$2B)",
    "Micro Cap ($50M-$300M)",
    # Index Based
    "S&P 500",
    "S&P 500 Top 50",
    "NASDAQ 100",
    "DOW 30",
    "Russell 2000 Top 100",
    "S&P MidCap 400 Top 50",
    "S&P SmallCap 600 Top 50",
    # Sector Based
    "S&P 500 Technology",
    "S&P 500 Healthcare",
    "S&P 500 Financials",
    "S&P 500 Consumer Discretionary",
    "S&P 500 Energy",
    # Thematic
    "NASDAQ Financial Top 10",
    "NYSE FANG+",
]

# S&P 500 (Full 500 stocks - Updated Dec 2024)
SP500 = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "GOOG", "NVDA", "META", "AVGO", "ADBE", "CRM", "AMD",
    "INTC", "CSCO", "TXN", "QCOM", "ORCL", "IBM", "NOW", "INTU", "AMAT", "MU",
    "ADI", "LRCX", "KLAC", "SNPS", "CDNS", "APH", "MSI", "FTNT", "PANW", "MCHP",
    "GLW", "KEYS", "HPQ", "HPE", "WDC", "STX", "NTAP", "ZBRA", "JNPR", "CDW",
    "AKAM", "FFIV", "VRSN", "ENPH", "MPWR", "SWKS", "QRVO", "TER", "SEDG", "GEN",
    # Healthcare
    "LLY", "UNH", "JNJ", "MRK", "ABBV", "PFE", "TMO", "DHR", "ABT", "BMY",
    "AMGN", "MDT", "ISRG", "GILD", "VRTX", "REGN", "CVS", "CI", "ELV", "SYK",
    "ZTS", "BDX", "BSX", "HCA", "MCK", "EW", "DXCM", "A", "IQV", "IDXX",
    "MTD", "WAT", "CAH", "COR", "HUM", "RMD", "ALGN", "WST", "HOLX", "MOH",
    "CNC", "VTRS", "TECH", "HSIC", "DGX", "LH", "PKI", "BIO", "BAX", "XRAY",
    # Financials
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "SPGI", "AXP",
    "BLK", "C", "SCHW", "CB", "MMC", "PGR", "USB", "CME", "AON", "PNC",
    "ICE", "MET", "AIG", "TRV", "AJG", "COF", "AFL", "PRU", "ALL", "MSCI",
    "BK", "TFC", "FI", "MCO", "WTW", "STT", "FIS", "CINF", "NDAQ", "RJF",
    "HBAN", "CFG", "KEY", "FITB", "RF", "NTRS", "SBNY", "ZION", "L", "RE",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "MAR",
    "CMG", "ORLY", "AZO", "DHI", "LEN", "F", "GM", "ROST", "YUM", "HLT",
    "DG", "DLTR", "EBAY", "ULTA", "POOL", "BBY", "APTV", "GPC", "LKQ", "GRMN",
    "PHM", "NVR", "TPR", "RL", "VFC", "PVH", "HAS", "WYNN", "LVS", "MGM",
    "DRI", "EXPE", "CCL", "RCL", "NCLH", "BWA", "LEG", "WHR", "MHK", "AAP",
    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "KMB",
    "EL", "GIS", "SYY", "K", "KHC", "HSY", "ADM", "STZ", "KDP", "MNST",
    "TSN", "CAG", "CHD", "CLX", "SJM", "MKC", "HRL", "CPB", "LW", "TAP",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX", "OXY", "PXD",
    "HES", "WMB", "KMI", "DVN", "HAL", "FANG", "BKR", "CTRA", "MRO", "OKE",
    "TRGP", "APA", "EQT", "PR", "OVV",
    # Industrials
    "CAT", "DE", "RTX", "UNP", "HON", "BA", "LMT", "GE", "UPS", "MMM",
    "FDX", "NOC", "ITW", "ETN", "CSX", "EMR", "NSC", "WM", "GD", "JCI",
    "PH", "PCAR", "CTAS", "TDG", "ODFL", "TT", "CARR", "ROK", "CMI", "DOV",
    "AME", "SWK", "XYL", "FAST", "HWM", "IR", "PWR", "WAB", "GWW", "ROP",
    "VRSK", "IEX", "MLM", "VMC", "J", "MAS", "SNA", "LII", "PAYX", "AOS",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "PEG",
    "WEC", "ES", "AWK", "DTE", "ETR", "FE", "PPL", "CMS", "CNP", "AEE",
    "NI", "EVRG", "ATO", "LNT", "NRG", "PNW",
    # Real Estate
    "AMT", "PLD", "CCI", "EQIX", "PSA", "WELL", "DLR", "O", "VICI", "SPG",
    "AVB", "EQR", "ARE", "VTR", "MAA", "UDR", "ESS", "SBAC", "WY", "EXR",
    "REG", "BXP", "CBRE", "HST", "KIM", "CPT", "IRM", "FRT", "AIV", "SLG",
    # Materials
    "LIN", "APD", "SHW", "ECL", "DD", "NUE", "FCX", "CTVA", "DOW", "PPG",
    "NEM", "VMC", "MLM", "IFF", "ALB", "FMC", "EMN", "CE", "CF", "MOS",
    "LYB", "BALL", "IP", "PKG", "WRK", "SEE", "AVY", "AMCR",
    # Communication Services
    "GOOGL", "GOOG", "META", "DIS", "CMCSA", "NFLX", "VZ", "T", "CHTR", "TMUS",
    "ATVI", "EA", "TTWO", "WBD", "PARA", "FOX", "FOXA", "NWS", "NWSA", "LYV",
    "OMC", "IPG", "MTCH", "DISH",
    # Additional S&P 500 constituents to reach ~500
    "FICO", "AXON", "ABNB", "CEG", "DECK", "EG", "ELV", "GEHC", "GEV", "HUBB",
    "INVH", "KVUE", "LULU", "ON", "OTIS", "PAYC", "PODD", "SMCI", "SOLV", "STLD",
    "TDY", "TSCO", "TTWO", "UAL", "URI", "VRSN", "WST", "ZBH",
    # Tech additions
    "ACGL", "ADP", "ANET", "ANSS", "BIIB", "BRO", "CBOE", "CDAY", "CPAY", "CPRT",
    "CTLT", "CTSH", "EPAM", "EQT", "ERIE", "EXPE", "FDS", "FLT", "FSLR", "GDDY",
    "GNRC", "INCY", "JKHY", "KMX", "LDOS", "LPLA", "LRCX", "LUV", "LVS", "LW",
    "LYV", "MOH", "MRNA", "NDAQ", "NDSN", "NVR", "ODFL", "PAYC", "PCAR", "PTC",
    "RVTY", "STE", "TER", "TFX", "TRMB", "TROW", "TRV", "TXT", "TYL", "UAL",
    "ULTA", "UNP", "URI", "VICI", "VRSN", "VTR", "VTRS", "WAB", "WAT", "WBA",
    "WDC", "WEC", "WELL", "WFC", "WHR", "WM", "WMB", "WRB", "WRK", "WST",
    "WTW", "WY", "WYNN", "XEL", "XOM", "XYL", "YUM", "ZBH", "ZBRA", "ZION", "ZTS"
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

# =============================================================================
# MARKET CAP BASED UNIVERSES
# =============================================================================

# Mega Cap (>$200B Market Cap)
MEGA_CAP = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "LLY", "WMT",
    "JPM", "V", "XOM", "MA", "JNJ", "ORCL", "COST", "HD", "AVGO", "PG",
    "NFLX", "BAC", "ABBV", "CRM", "MRK", "CVX", "AMD", "PLTR", "KO", "PEP"
]

# Large Cap ($10B-$200B Market Cap)
LARGE_CAP = [
    "UNH", "CSCO", "TMO", "ABT", "MCD", "ACN", "DHR", "ADBE", "TXN", "QCOM",
    "INTC", "VZ", "WFC", "CMCSA", "PM", "NEE", "BA", "RTX", "UNP", "HON",
    "IBM", "AMGN", "LOW", "SBUX", "GE", "CAT", "BKNG", "PLD", "SPGI", "BLK",
    "GS", "MS", "ISRG", "VRTX", "GILD", "MMM", "AXP", "SYK", "MDT", "REGN",
    "ADP", "SCHW", "TJX", "CB", "DE", "LMT", "CI", "SO", "DUK", "ZTS"
]

# Medium Cap ($2B-$10B Market Cap)
MEDIUM_CAP = [
    "SNAP", "ROKU", "PINS", "ETSY", "RBLX", "U", "DKNG", "COIN", "HOOD", "AFRM",
    "SQ", "SHOP", "ABNB", "DASH", "LYFT", "UBER", "RIVN", "LCID", "NIO", "XPEV",
    "CHWY", "PTON", "FVRR", "UPWK", "ZM", "DOCU", "CRWD", "OKTA", "NET", "FSLY",
    "DDOG", "SNOW", "MDB", "BILL", "HUBS", "TWLO", "ZS", "PANW", "FTNT", "TTD",
    "MNST", "POOL", "WSM", "DPZ", "CMG", "WING", "TXRH", "EAT", "DRI", "BLMN"
]

# Small Cap ($300M-$2B Market Cap)
SMALL_CAP = [
    "SOFI", "UPST", "CLOV", "PLTR", "MARA", "RIOT", "BITF", "CLSK", "CIFR", "HUT",
    "PLUG", "FCEL", "BLDP", "BE", "CHPT", "BLNK", "EVGO", "DCFC", "NKLA", "GOEV",
    "SPCE", "RKLB", "ASTR", "LUNR", "RDW", "ASTS", "GSAT", "IRDM", "VSAT", "GOGO",
    "AMC", "GME", "BBBY", "EXPR", "NAKD", "SNDL", "ACB", "TLRY", "CGC", "CRON",
    "CRSP", "NTLA", "EDIT", "BEAM", "VERV", "MRNA", "BNTX", "NVAX", "OCGN", "VXRT"
]

# Micro Cap ($50M-$300M Market Cap)
MICRO_CAP = [
    "MULN", "FFIE", "WKHS", "RIDE", "FSR", "ARVL", "REE", "PSNY", "PTRA", "XOS",
    "HYLN", "NKLA", "LEV", "ELMS", "GOEV", "SOLO", "AYRO", "WKSP", "CENN", "FREY",
    "VUZI", "MVIS", "LAZR", "LIDR", "VLDR", "INVZ", "OUST", "CPTN", "AEVA", "AEYE",
    "BYND", "TTCF", "APPH", "AGFY", "VFF", "GRWG", "HYFM", "SMG", "STKL", "VITL",
    "SKLZ", "SLGG", "ESPO", "HEAR", "CRSR", "LOGI", "GPRO", "SONO", "KOSS", "VOXX"
]

# Universe dictionary
UNIVERSES = {name: [] for name in INDEX_NAMES}

# Populate universes - Market Cap Based
UNIVERSES["Mega Cap (>$200B)"] = MEGA_CAP
UNIVERSES["Large Cap ($10B-$200B)"] = LARGE_CAP
UNIVERSES["Medium Cap ($2B-$10B)"] = MEDIUM_CAP
UNIVERSES["Small Cap ($300M-$2B)"] = SMALL_CAP
UNIVERSES["Micro Cap ($50M-$300M)"] = MICRO_CAP

# Populate universes - Index Based
UNIVERSES["S&P 500"] = SP500
UNIVERSES["S&P 500 Top 50"] = SP500_TOP_50
UNIVERSES["NASDAQ 100"] = NASDAQ_100
UNIVERSES["DOW 30"] = DOW_30
UNIVERSES["Russell 2000 Top 100"] = RUSSELL_2000_SAMPLE
UNIVERSES["S&P MidCap 400 Top 50"] = SP_MIDCAP_400_SAMPLE
UNIVERSES["S&P SmallCap 600 Top 50"] = SP_SMALLCAP_600_SAMPLE

# Populate universes - Sector Based
UNIVERSES["S&P 500 Technology"] = SP500_TECH
UNIVERSES["S&P 500 Healthcare"] = SP500_HEALTHCARE
UNIVERSES["S&P 500 Financials"] = SP500_FINANCIALS
UNIVERSES["S&P 500 Consumer Discretionary"] = SP500_CONSUMER_DISC
UNIVERSES["S&P 500 Energy"] = SP500_ENERGY

# Populate universes - Thematic
UNIVERSES["NASDAQ Financial Top 10"] = ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "BK", "SCHW"]
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
        "S&P 500 Top 50", "NASDAQ 100", "DOW 30", "Russell 2000 Top 100",
        "S&P MidCap 400 Top 50", "S&P SmallCap 600 Top 50"
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
        "S&P 500 Top 50", "S&P MidCap 400 Top 50", "S&P SmallCap 600 Top 50", "Russell 2000 Top 100"
    ]


def get_thematic_universes():
    """Get strategy/thematic indices."""
    return [
        "NYSE FANG+", "NASDAQ Financial"
    ]
