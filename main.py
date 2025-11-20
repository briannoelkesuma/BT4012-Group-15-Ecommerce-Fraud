import json

# REPLACE 'your_notebook.ipynb' with your actual filename
filename = 'ecommerce_fraud_detection_pipeline.ipynb'

try:
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # This is the specific fix for the GitHub error
    if 'metadata' in data and 'widgets' in data['metadata']:
        print("Found corrupted widget metadata. Removing it...")
        del data['metadata']['widgets']
        
        # Save the fixed version back to the same file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
            
        print("Success! The file is fixed.")
    else:
        print("No widget metadata found. The file might already be clean.")

except Exception as e:
    print(f"Error: {e}")