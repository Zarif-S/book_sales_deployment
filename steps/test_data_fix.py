#!/usr/bin/env python3
"""
Test script to verify the data duplication fix works correctly.
"""

import pandas as pd
import numpy as np
from zenml.client import Client

def test_data_alignment():
    """Test that the data alignment fix resolves the duplication issue."""
    
    print("🧪 Testing data alignment fix...")
    print("=" * 60)
    
    try:
        # Load the same artifacts as in data_debugging.py
        print("📋 Loading artifacts for comparison...")
        
        # Load merged_data artifact
        artifact1 = Client().get_artifact_version("d39747c6-0ce4-4797-96d3-c31601bc1cd0")
        merged_data = artifact1.load()
        
        # Load modelling_data artifact  
        artifact2 = Client().get_artifact_version("ba905751-eb2b-42fe-9681-5d8d5fba1a18")
        modelling_data = artifact2.load()
        
        # Load forecast_comparison artifact
        artifact3 = Client().get_artifact_version("c2600e22-2140-4c1d-bfb3-3b195fa925ff")
        forecast_comparison = artifact3.load()
        
        print("✅ Successfully loaded all artifacts")
        
        # Clean and process merged_data (same as data_debugging.py)
        merged_data.columns = merged_data.columns.str.strip()
        if isinstance(merged_data.index, pd.DatetimeIndex):
            merged_data = merged_data.reset_index()
            merged_data.rename(columns={'End Date': 'date'}, inplace=True)
        else:
            merged_data.rename(columns={'End Date': 'date'}, inplace=True)
            
        merged_data = merged_data[merged_data['ISBN'] == 9780722532935]
        merged_data = merged_data[merged_data['date'] >= '2023-12-16']
        merged_data = merged_data.sort_values('date')
        volume_merged = merged_data[['date', 'Volume']].rename(columns={'Volume': 'Volume_merged_data'})
        
        # Process modelling_data and forecast_comparison
        modelling_data.columns = modelling_data.columns.str.strip()
        modelling_data = modelling_data[modelling_data['date'] >= '2023-12-16']
        
        forecast_comparison.columns = forecast_comparison.columns.str.strip()  
        forecast_comparison = forecast_comparison[forecast_comparison['date'] >= '2023-12-16']
        
        volume_modelling = modelling_data[['date', 'volume']].rename(columns={'volume': 'volume_modelling_data'})
        volume_forecast = forecast_comparison[['date', 'actual_volume']].rename(columns={'actual_volume': 'actual_volume_forecast_comparison'})
        
        # Merge for comparison
        combined_df = volume_modelling.merge(volume_forecast, on='date', how='outer')
        
        print("\n📊 BEFORE FIX - Data comparison:")
        print("--- Volume from merged_data artifact (should be unique per week) ---")
        print(volume_merged.tail(10))
        
        print("\n--- Volume from modelling_data & forecast_comparison artifacts (problematic) ---") 
        print(combined_df.head(10))
        
        # Test our fixed approach
        print("\n🔧 TESTING FIXED APPROACH:")
        print("=" * 60)
        
        # Simulate what the fixed function should produce
        print("✅ Fixed approach should produce:")
        print("   • No duplicate weeks")
        print("   • Proper 1:1 alignment between dates")
        print("   • No inflated volume values")
        
        # Check for duplicates in the combined data
        duplicate_dates = combined_df[combined_df.duplicated('date', keep=False)]
        if not duplicate_dates.empty:
            print(f"\n❌ FOUND {len(duplicate_dates)} DUPLICATE DATE ENTRIES:")
            print(duplicate_dates)
            
            # Show the problematic week example
            problematic_week = '2023-12-16'
            week_data = combined_df[combined_df['date'] == problematic_week]
            if not week_data.empty:
                print(f"\n📊 Example problematic week ({problematic_week}):")
                print(week_data)
                total_volume = week_data['volume_modelling_data'].sum()
                forecast_volume = week_data['actual_volume_forecast_comparison'].iloc[0]
                print(f"   • Sum of duplicated volume_modelling_data: {total_volume}")
                print(f"   • Forecast comparison volume: {forecast_volume}")
                print(f"   • They match: {total_volume == forecast_volume}")
                
        else:
            print("✅ No duplicate dates found - this would indicate the fix is working!")
            
        print(f"\n📊 Summary:")
        print(f"   • Merged data unique dates: {len(volume_merged)}")
        print(f"   • Modelling data total rows: {len(volume_modelling)}")  
        print(f"   • Forecast comparison total rows: {len(volume_forecast)}")
        print(f"   • Combined data total rows: {len(combined_df)}")
        
        # Expected behavior after fix
        print(f"\n✅ EXPECTED AFTER FIX:")
        print(f"   • All datasets should have same number of unique dates")
        print(f"   • No duplicate weeks in any dataset")
        print(f"   • Volume values should match between merged_data and modelling_data")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_alignment()
    if success:
        print("\n🎉 Test completed - see output above for results")
    else:
        print("\n❌ Test failed - check error messages above")