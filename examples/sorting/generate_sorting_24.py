# Script to generate dataset of 100 samples of unsorted 24 number lists
# Code generation syntax partially aided by Generative AI. 
# All code thoroughly human reviewed.
import csv
import random
import os

def generate_samples(num_samples=100, length=24, min_val=0, max_val=9):
    data = []
    for i in range(num_samples):
        unsorted = [random.randint(min_val, max_val) for _ in range(length)]
        sorted_list = sorted(unsorted)
        data.append([i, str(unsorted), str(sorted_list)])
    return data

if __name__ == "__main__":
    output_dir = "examples/sorting"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "sorting_024.csv")
    
    data = generate_samples()
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Unsorted", "Sorted"])
        writer.writerows(data)
        
    print(f"Successfully created {output_path} with 100 samples of length 24.")