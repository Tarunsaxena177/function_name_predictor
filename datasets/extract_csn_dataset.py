from datasets import load_dataset
import csv
import re
import os
import sys

def extract_params(code):
    match = re.search(r'def\s+\w+\s*\((.*?)\):', code, re.DOTALL)
    if match:
        params = re.sub(r'\s+', ' ', match.group(1))
        return params.replace(',', ';').strip()
    return ""

def clean_text(text):
    if not text: return ""
    return " ".join(text.split()).replace(',', ' ').strip()

def prepare_dataset():
    print("Downloading CodeSearchNet Python subset...")
    
    # Use try/except to ensure we can handle cleanup
    try:
        dataset = load_dataset("code_search_net", "python", split="train", streaming=True)
        
        output_file = 'dataset.csv'
        header = ['description', 'parameters', 'return_type', 'library', 'keywords', 'param_count', 'function_name']
        
        count = 0
        max_samples = 5000
        
        print(f"Processing and saving to {output_file}...")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for entry in dataset:
                if count >= max_samples:
                    break
                    
                description = clean_text(entry.get('func_documentation_string', ''))
                function_name = entry.get('func_name', '')
                code = entry.get('func_code_string', '')
                
                if not description or not function_name or function_name.startswith('_'):
                    continue
                    
                parameters = extract_params(code)
                repository = entry.get('repository_name', 'unknown/unknown')
                library = repository.split('/')[-1]
                keywords = " ".join(description.split()[:5]).lower()
                param_count = len([p for p in parameters.split(';') if p.strip()])
                
                writer.writerow([description, parameters, "Any", library, keywords, param_count, function_name])
                
                count += 1
                if count % 500 == 0:
                    print(f"Extracted {count} samples...")

        print(f"Successfully generated {output_file} with {count} samples.")
        
        # Explicitly delete the dataset to close background threads
        del dataset

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    prepare_dataset()
    print("Exiting...")
    # os._exit(0) is a blunt tool that stops the process immediately 
    # without waiting for the background threads that cause the GIL error.
    os._exit(0)