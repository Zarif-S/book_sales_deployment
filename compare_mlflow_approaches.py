#!/usr/bin/env python3

"""
Compare the old vs new MLflow tracking approaches for scalability analysis
"""

import mlflow
from datetime import datetime

def compare_approaches():
    client = mlflow.MlflowClient()
    
    print("=" * 80)
    print("MLFLOW TRACKING APPROACHES COMPARISON")
    print("=" * 80)
    
    # Old Approach - Prefixed metrics in single run
    print("\n📊 OLD APPROACH (Single Run with Prefixed Metrics)")
    print("-" * 60)
    
    try:
        old_runs = client.search_runs(["394282808953726458"], max_results=1)  # book_sales_arima_modeling_pipeline
        if old_runs:
            old_run = old_runs[0]
            print(f"🔄 Run: {old_run.info.run_name}")
            print(f"   Status: {old_run.info.status}")
            print(f"   Total Metrics: {len(old_run.data.metrics)}")
            
            # Show book-specific metrics (prefixed)
            book_metrics = [m for m in old_run.data.metrics.keys() if 'book_' in m]
            print(f"   Book-specific metrics: {len(book_metrics)}")
            print("   Sample metrics:")
            for metric in book_metrics[:5]:
                print(f"     • {metric}: {old_run.data.metrics[metric]}")
                
        print(f"\n💡 Scalability Analysis (Old Approach):")
        print(f"   • With 2 books: ~14 metrics")
        print(f"   • With 10 books: ~70 metrics (cluttered)")
        print(f"   • With 50 books: ~350 metrics (unusable)")
        print(f"   • With 100 books: ~700 metrics (disaster)")
        
    except Exception as e:
        print(f"Error accessing old approach data: {e}")
    
    # New Approach - Hybrid with individual runs
    print(f"\n🚀 NEW APPROACH (Hybrid: Parent + Individual Child Runs)")
    print("-" * 60)
    
    try:
        # Parent run
        parent_runs = client.search_runs(["394282808953726458"], max_results=1)
        if parent_runs:
            parent_run = parent_runs[0]
            print(f"📈 Parent Run: {parent_run.info.run_name}")
            print(f"   Status: {parent_run.info.status}")
            print(f"   Pipeline Metrics: {len(parent_run.data.metrics)}")
            print(f"   Sample pipeline metrics:")
            for metric, value in list(parent_run.data.metrics.items())[:4]:
                print(f"     • {metric}: {value}")
        
        # Individual book runs
        book_runs = client.search_runs(["367421319805540949"], max_results=5)  # book_sales_arima_modeling_v2
        print(f"\n📖 Individual Book Runs: {len(book_runs)}")
        for run in book_runs:
            print(f"   🔄 {run.info.run_name}")
            print(f"      Status: {run.info.status}")
            print(f"      Clean Metrics: {len(run.data.metrics)}")
            print(f"      Clean Parameters: {list(run.data.params.keys())[:5]}")
        
        print(f"\n💡 Scalability Analysis (New Approach):")
        print(f"   • With 2 books: 1 parent + 2 child runs (organized)")
        print(f"   • With 10 books: 1 parent + 10 child runs (manageable)")
        print(f"   • With 50 books: 1 parent + 50 child runs (scalable)")
        print(f"   • With 100 books: 1 parent + 100 child runs (perfect)")
        
    except Exception as e:
        print(f"Error accessing new approach data: {e}")
    
    print(f"\n🎯 BENEFITS OF NEW APPROACH:")
    print(f"   ✅ Clean, comparable metrics per book")
    print(f"   ✅ Easy filtering and searching")
    print(f"   ✅ Individual book performance tracking")
    print(f"   ✅ Pipeline-level overview maintained") 
    print(f"   ✅ Scales to hundreds of books")
    print(f"   ✅ Better MLflow UI experience")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    compare_approaches()