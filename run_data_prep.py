import os
from pipeline.catalog_parser import CatalogDataEngine

def main():
    # Directories verify karein
    os.makedirs("output", exist_ok=True)
    
    # Paths define karein (Aapne jis naam se data save kiya ho use adjust kar sakte hain)
    raw_catalog_path = "data/tic_ctl_crossmatched.csv"  # Check your actual filename inside data/
    output_index_path = "output/filtered_index.csv"
    
    if not os.path.exists(raw_catalog_path):
        # Agar compressed form (.gz) mein hai, toh direct path pass karein (DuckDB reads .gz automatically!)
        if os.path.exists(raw_catalog_path + ".gz"):
            raw_catalog_path += ".gz"
        else:
            print(f"❌ Error: Please ensure your 9.5 GB file is located at '{raw_catalog_path}'")
            return

    # Engine initialize aur execute karein
    try:
        engine = CatalogDataEngine()
        engine.filter_and_create_index(raw_catalog_path, output_index_path)
    except Exception as e:
        print(f"❌ Execution failed: {str(e)}")

if __name__ == "__main__":
    main()