import os
import duckdb
from pipeline.data_models import StellarParameters, ValidationError

class CatalogDataEngine:
    def __init__(self, db_path: str = ":memory:"):
        """Initializes DuckDB in-memory session for ultra-fast processing."""
        self.conn = duckdb.connect(db_path)

    def filter_and_create_index(self, input_csv_path: str, output_csv_path: str):
        """
        Forces DuckDB to parse using the exact three-digit zero-padded column name structure
        revealed by the engine's compilation logs.
        """
        if not os.path.exists(input_csv_path):
            raise FileNotFoundError(f"Raw catalog not found at {input_csv_path}")
        
        print(f"🚀 Blazing fast execution: Filtering {input_csv_path} via DuckDB...")
        
        # Exact 3-digit mapping matching your system trace:
        # column000=ID, column070=rad, column072=mass, column064=teff, column108=tmag, column066=logg
        query = f"""
            COPY (
                SELECT 
                    CAST(column000 AS VARCHAR) AS ID, 
                    CAST(column070 AS DOUBLE) AS rad, 
                    CAST(column072 AS DOUBLE) AS mass, 
                    CAST(column064 AS DOUBLE) AS teff, 
                    CAST(column108 AS DOUBLE) AS tmag, 
                    CAST(column066 AS DOUBLE) AS logg
                FROM read_csv('{input_csv_path}', 
                             header=False, 
                             delim=',', 
                             auto_detect=True)
                WHERE TRY_CAST(column108 AS DOUBLE) <= 13.5
                  AND TRY_CAST(column064 AS DOUBLE) BETWEEN 4000.0 AND 7000.0
                  AND column070 IS NOT NULL
                  AND column072 IS NOT NULL
                  AND TRY_CAST(column066 AS DOUBLE) >= 3.5
            ) TO '{output_csv_path}' (HEADER, DELIMITER ',');
        """
        self.conn.execute(query)
        print(f"✅ Filtered index successfully created at: {output_csv_path}")

    def get_stellar_metadata(self, target_id: str, filtered_index_path: str) -> StellarParameters:
        """Quickly fetches metadata for a specific TIC ID from the filtered index."""
        clean_id = target_id.replace("TIC", "").strip()
        
        query = f"""
            SELECT ID, rad, mass, teff, logg FROM '{filtered_index_path}' 
            WHERE ID = '{clean_id}' LIMIT 1;
        """
        result = self.conn.execute(query).fetchone()
        
        if not result:
            raise ValidationError(f"Target ID {target_id} not found in filtered index.")
            
        return StellarParameters(
            tic_id=f"TIC {result[0]}",
            stellar_radius=float(result[1]),
            stellar_mass=float(result[2]),
            teff=float(result[3]),
            logg=float(result[4]),
            metallicity=0.0
        )