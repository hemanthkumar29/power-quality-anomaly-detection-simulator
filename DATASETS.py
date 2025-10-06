"""
Dataset information and sources for Power Quality analysis
"""

DATASET_SOURCES = {
    "synthetic": {
        "name": "Built-in Synthetic Generator",
        "description": "Generates PQ waveforms based on IEEE standards",
        "license": "MIT (Generated data)",
        "classes": ["Normal", "Sag", "Swell", "Harmonic", "Outage"],
        "configurable": True,
        "url": None
    },
    
    "uci": {
        "name": "UCI Machine Learning Repository - Power Quality",
        "description": "Power quality disturbance dataset",
        "license": "Check UCI repository",
        "classes": "Various PQ events",
        "url": "https://archive.ics.uci.edu/ml/datasets.php",
        "search_terms": ["power quality", "electrical", "voltage"]
    },
    
    "ieee_dataport": {
        "name": "IEEE DataPort - Power Quality Datasets",
        "description": "Various power quality monitoring datasets",
        "license": "Varies by dataset",
        "url": "https://ieee-dataport.org/",
        "search_terms": ["power quality", "voltage sag", "harmonics", "PQ events"]
    },
    
    "kaggle": {
        "name": "Kaggle - Power System Datasets",
        "description": "Community-contributed power quality datasets",
        "license": "Varies by dataset",
        "url": "https://www.kaggle.com/datasets",
        "search_terms": ["power quality", "electrical waveform", "voltage disturbance"]
    },
    
    "github": {
        "name": "GitHub - Research Datasets",
        "description": "Academic research datasets on power quality",
        "license": "Varies by repository",
        "url": "https://github.com/search",
        "search_terms": ["power quality dataset", "PQ waveform", "electrical disturbance"]
    }
}

STANDARDS = {
    "IEEE_1159": {
        "name": "IEEE 1159-2019",
        "description": "Recommended Practice for Monitoring Electric Power Quality",
        "categories": [
            "Transients",
            "Short-duration variations (sag, swell, interruption)",
            "Long-duration variations",
            "Voltage imbalance",
            "Waveform distortion (harmonics)",
            "Voltage fluctuations",
            "Power frequency variations"
        ]
    },
    
    "IEC_61000": {
        "name": "IEC 61000 Series",
        "description": "Electromagnetic Compatibility (EMC) Standards",
        "focus": "Power quality limits and testing methods"
    }
}

def print_dataset_info():
    """Print information about available datasets"""
    print("\n" + "="*70)
    print("POWER QUALITY DATASETS - SOURCES AND INFORMATION")
    print("="*70)
    
    for key, info in DATASET_SOURCES.items():
        print(f"\n{info['name']}")
        print("-" * 70)
        print(f"Description: {info['description']}")
        print(f"License: {info.get('license', 'N/A')}")
        if info.get('url'):
            print(f"URL: {info['url']}")
        if info.get('search_terms'):
            print(f"Search terms: {', '.join(info['search_terms'])}")
    
    print("\n" + "="*70)
    print("RELEVANT STANDARDS")
    print("="*70)
    
    for key, standard in STANDARDS.items():
        print(f"\n{standard['name']}")
        print(f"Description: {standard['description']}")
        if 'categories' in standard:
            print("Categories:")
            for cat in standard['categories']:
                print(f"  - {cat}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    print_dataset_info()
