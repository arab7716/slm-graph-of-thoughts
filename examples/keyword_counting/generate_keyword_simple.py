# Script to generate simplified keyword counting dataset
# Code generation syntax partially aided by Generative AI. 
# All code thoroughly human reviewed.

import csv
import random
import os
import re

# config for dataset
NUM_SAMPLES = 100
MIN_SENTENCES = 3
MAX_SENTENCES = 6
COUNTRIES = [
    "USA", "Canada", "Mexico", "Brazil", "Argentina", 
    "France", "Germany", "Italy", "Spain", "UK", 
    "China", "Japan", "India", "Russia", "Australia",
    "Egypt", "Kenya", "Peru", "Chile", "Vietnam"
]

TEMPLATES = [
    "I traveled to {c1} and then visited {c2}.",
    "The delegation from {c1} met with {c2}.",
    "{c1} is famous for its food, unlike {c2}.",
    "From {c1}, we flew to {c2} and finally {c3}.",
    "My friend from {c1} loves {c2}.",
    "We saw {c1} and {c2} on the map.",
    "Both {c1} and {c2} are beautiful.",
    "After leaving {c1}, I went to {c1} again." # duplicate case
]

def generate_samples():
    data = []
    for i in range(NUM_SAMPLES):
        num_sentences_to_gen = random.randint(MIN_SENTENCES, MAX_SENTENCES)
        sentences = []
        
        for _ in range(num_sentences_to_gen):
            tpl = random.choice(TEMPLATES)
            needed = tpl.count("{c")
            
            # Pick random countries
            picks = random.choices(COUNTRIES, k=needed)
            
            formatted = tpl
            for j, country in enumerate(picks):
                formatted = formatted.replace(f"{{c{j+1}}}", country, 1)
                formatted = formatted.replace(f"{{c{j+1}}}", country) 
            
            sentences.append(formatted)
            
        full_text = " ".join(sentences)
        

        all_countries_in_text = []
        country_pattern = r'\b(' + '|'.join(re.escape(c) for c in COUNTRIES) + r')\b'
        
        found_countries = re.findall(country_pattern, full_text)
        all_countries_in_text = found_countries
        
        ground_truth_str = "[" + ", ".join(all_countries_in_text) + "]"
        
        sentence_count = len(sentences)
        char_count = len(full_text)
        
        data.append([i, full_text, ground_truth_str, sentence_count, char_count])
    return data

if __name__ == "__main__":
    output_dir = "keyword_counting"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "countries_simple.csv")
    
    data = generate_samples()
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Text", "Countries", "Sentences", "Characters"])
        writer.writerows(data)
        
    print(f"Successfully created {output_path}")
    print("Format matches original countries.csv (5 columns, unquoted lists).")