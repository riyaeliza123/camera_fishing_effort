"""DataFrame construction and management."""

import pandas as pd


def build_dataframe_row(idx: int, filename: str, location: str, image_date: str, 
                       image_time: str, model_type: str, in_count: int = None, 
                       out_count: int = None, boat_count: int = None) -> dict:
    """Build a single DataFrame row based on model type.
    
    Args:
        idx: Row index
        filename: Image filename
        location: Location string
        image_date: Date from EXIF
        image_time: Time from EXIF
        model_type: "chokepoint" or "fishing"
        in_count: Count of objects going in (chokepoint only)
        out_count: Count of objects going out (chokepoint only)
        boat_count: Count of boats (fishing only)
        
    Returns:
        Dictionary representing a DataFrame row
    """
    base_row = {
        "Sl No": idx,
        "Image Name": filename,
        "Location": location,
        "Date": image_date,
        "Time": image_time,
    }
    
    if model_type == "chokepoint":
        return {**base_row, "In": in_count, "Out": out_count}
    else:  # fishing
        return {**base_row, "Boat": boat_count}


def create_dataframe(df_data: list) -> pd.DataFrame:
    """Create a DataFrame from list of row dictionaries.
    
    Args:
        df_data: List of dictionaries from build_dataframe_row()
        
    Returns:
        pandas DataFrame
    """
    return pd.DataFrame(df_data)
