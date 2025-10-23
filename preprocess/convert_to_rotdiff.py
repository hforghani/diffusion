import json
import sys
import os
import glob

def merge_graph_files(input_dir, output_dir):
    # Check if graph.txt exists
    default_graph = os.path.join(input_dir, 'graph.txt')
    if os.path.exists(default_graph):
        input_graph = default_graph
    else:
        # Get all txt files that might contain graph data
        graph_files = glob.glob(os.path.join(input_dir, '*graph*.txt'))
        if not graph_files:
            print("No graph files found in input directory")
            return
        
        # Get the largest file
        input_graph = max(graph_files, key=os.path.getsize)
        print(f"Using largest graph file: {os.path.basename(input_graph)}")
    
    # Create output edges file
    output_file = os.path.join(output_dir, 'edges.txt')
    with open(output_file, 'w') as outf:
        with open(input_graph, 'r') as inf:
            for line in inf:
                outf.write(line)

def convert_sequences_to_cascades(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the JSON file
    input_file = os.path.join(input_dir, 'sequences.json')
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Open output file
    output_file = os.path.join(output_dir, 'cascades.txt')
    with open(output_file, 'w') as f:
        # Process each cascade
        for user_id, info in data.items():
            # Skip empty cascades
            if not info['cascade']:
                continue
                
            # Create cascade line
            cascade_points = []
            for i, point in enumerate(info['cascade']):
                # Each point is a list with one value
                timestamp = point[0]
                cascade_points.append(f"{user_id} {timestamp}")
            
            # Write cascade line
            if cascade_points:
                f.write(', '.join(cascade_points) + '\n')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convert_to_rotdiff.py <input_dir> <output_dir>")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Merge graph files first
    merge_graph_files(input_dir, output_dir)
    
    # Then convert sequences to cascades
    convert_sequences_to_cascades(input_dir, output_dir)