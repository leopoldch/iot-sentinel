CANONICAL_FAMILY_CLASSES = [
    "benign",
    "ddos_dos",
    "recon",
    "spoofing_mitm",
    "credential",
    "web_injection",
    "malware",
]


def lookup(value, mapping, dataset_name, label):
    if value in mapping:
        return mapping[value]
    raise ValueError(f"Unknown {dataset_name} attack label: {label}")


def map_ciciot_family(label):
    value = str(label).strip().upper()
    if value == "BENIGN":
        return "benign"
    if value.startswith(("DDOS-", "DOS-", "MIRAI-")):
        return "ddos_dos"
    if value.startswith("RECON-") or value == "VULNERABILITYSCAN":
        return "recon"
    return lookup(
        value,
        {
            "DNS_SPOOFING": "spoofing_mitm",
            "MITM-ARPSPOOFING": "spoofing_mitm",
            "DICTIONARYBRUTEFORCE": "credential",
            "BROWSERHIJACKING": "web_injection",
            "COMMANDINJECTION": "web_injection",
            "SQLINJECTION": "web_injection",
            "UPLOADING_ATTACK": "web_injection",
            "XSS": "web_injection",
            "BACKDOOR_MALWARE": "malware",
        },
        "CICIoT2023",
        label,
    )


def map_edge_family(label):
    value = str(label).strip()
    if value == "Normal":
        return "benign"
    if value.startswith("DDoS_"):
        return "ddos_dos"
    return lookup(
        value,
        {
            "Fingerprinting": "recon",
            "Port_Scanning": "recon",
            "Vulnerability_scanner": "recon",
            "MITM": "spoofing_mitm",
            "Password": "credential",
            "SQL_injection": "web_injection",
            "Uploading": "web_injection",
            "XSS": "web_injection",
            "Backdoor": "malware",
            "Ransomware": "malware",
        },
        "Edge-IIoTset",
        label,
    )


def map_ton_family(label):
    value = str(label).strip().lower()
    return lookup(
        value,
        {
            "normal": "benign",
            "ddos": "ddos_dos",
            "scanning": "recon",
            "password": "credential",
            "injection": "web_injection",
            "xss": "web_injection",
            "backdoor": "malware",
            "ransomware": "malware",
        },
        "TON-IoT",
        label,
    )
