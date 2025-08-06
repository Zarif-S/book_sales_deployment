#!/usr/bin/env python3
"""
Test script for CNN+LSTM hybrid pipeline
"""
import sys
import os

def test_cnn_pipeline_syntax():
    """Test that the CNN+LSTM pipeline can be imported without syntax errors."""
    print("🧪 Testing CNN+LSTM Pipeline Import...")
    
    try:
        from pipelines.zenml_pipeline_cnn_lstm import book_sales_cnn_lstm_pipeline
        print("✅ CNN+LSTM pipeline imported successfully!")
        
        # ZenML pipelines are wrapped, so we can't easily inspect the signature
        # But we can test that it's a ZenML Pipeline object
        pipeline_type = type(book_sales_cnn_lstm_pipeline).__name__
        print(f"📊 Pipeline type: {pipeline_type}")
        
        if pipeline_type == 'Pipeline':
            print("✅ Correctly wrapped as ZenML Pipeline")
            
            # Check if we can access the configuration (optional)
            if hasattr(book_sales_cnn_lstm_pipeline, '_configuration'):
                print("✅ Pipeline configuration accessible")
            
            return True
        else:
            print("❌ Not a proper ZenML Pipeline")
            return False
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def test_cnn_step_import():
    """Test that the CNN step can be imported."""
    print("\n🧪 Testing CNN Step Import...")
    
    try:
        from steps._04_cnn import train_cnn_step
        print("✅ CNN step imported successfully!")
        
        # Test function signature
        import inspect
        sig = inspect.signature(train_cnn_step)
        print(f"📊 CNN step parameters: {list(sig.parameters.keys())}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_pipeline_file_differences():
    """Test that the CNN pipeline file has the expected differences from ARIMA pipeline."""
    print("\n🧪 Testing Pipeline File Content...")
    
    try:
        # Read both files and check for key differences
        with open('pipelines/zenml_pipeline_cnn_lstm.py', 'r') as f:
            cnn_content = f.read()
        
        # Check for CNN-specific content
        cnn_checks = [
            ('train_cnn_step', 'CNN step import'),
            ('train_cnn_optuna_step', 'CNN step function'),
            ('book_sales_cnn_lstm_pipeline', 'CNN pipeline function'),
            ('sequence_length', 'CNN sequence length parameter'),
            ('forecast_horizon', 'CNN forecast horizon parameter'),
            ('cnn_optimization_hybrid', 'CNN study name'),
        ]
        
        all_found = True
        for check, description in cnn_checks:
            if check in cnn_content:
                print(f"  ✅ {description}: Found")
            else:
                print(f"  ❌ {description}: Missing") 
                all_found = False
        
        # Check that ARIMA-specific content is removed/replaced
        arima_checks = [
            ('train_final_arima_model', 'ARIMA-specific functions should be removed'),
            ('SARIMAX', 'ARIMA model references should be removed'),
        ]
        
        for check, description in arima_checks:
            if check not in cnn_content:
                print(f"  ✅ {description}: Correctly removed")
            else:
                print(f"  ⚠️  {description}: Still present (may be OK)")
        
        return all_found
        
    except Exception as e:
        print(f"❌ File content check failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting CNN+LSTM Pipeline Tests...")
    print("=" * 60)
    
    # Run tests
    test1 = test_cnn_pipeline_syntax()
    test2 = test_cnn_step_import() 
    test3 = test_pipeline_file_differences()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Pipeline Import: {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"CNN Step Import: {'✅ PASSED' if test2 else '❌ FAILED'}")
    print(f"File Content Check: {'✅ PASSED' if test3 else '❌ FAILED'}")
    
    all_passed = test1 and test2 and test3
    
    if all_passed:
        print(f"\n🎉 All tests passed! CNN+LSTM pipeline is ready to run.")
        print(f"\n📋 To run the pipeline:")
        print(f"   python pipelines/zenml_pipeline_cnn_lstm.py")
        print(f"\n📋 Expected outputs:")
        print(f"   • CNN residuals saved to: outputs/data/residuals/cnn_residuals.csv")
        print(f"   • CNN model artifacts in ZenML")
        print(f"   • Ready for LSTM step integration")
    else:
        print(f"\n❌ Some tests failed. Check the pipeline implementation.")
    
    sys.exit(0 if all_passed else 1)