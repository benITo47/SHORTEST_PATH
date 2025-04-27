# Project Name

## Description
This project visualizes algorithms in graph theory and their step-by-step execution. The steps are generated as images, and a video can be created from those images using `ffmpeg`. Created for my own use to generate proper visualizations for PowerPoints. 

## Requirements
To run this project, you need to have the following Python libraries installed:

- **NetworkX** – for working with graphs
- **Matplotlib** – for visualizing the graphs
- **NumPy** – for numerical operations

### Installation

To install the required libraries, run the following command:

```bash
pip install networkx matplotlib numpy
```
## Running the Algorithms

To run the algorithms, simply execute the following Python scripts:

- `astar.py`: Implements the A* algorithm for finding the shortest path in a graph.
- `Astart_diff.py`: A variant of the A* algorithm with **ridiculous heuristics**, which is more experimental.
- `bellman_ford.py`: Implements the Bellman-Ford algorithm for finding the shortest path, even with negative edge weights.
- `Dijkstra.py`: Implements the Dijkstra algorithm for finding the shortest path with non-negative edge weights.
- `Floyd_warshall.py`: Implements the Floyd-Warshall algorithm for finding the shortest paths between all pairs of nodes in a graph.

To run any of these algorithms, simply call the respective Python script.


## Video Generation
After generating the images for each step of the algorithm (saved as PNG files), you can create a video from these images using `ffmpeg`.

### FFmpeg Command
To generate a video from the images, run the following `ffmpeg` command:

```bash
ffmpeg -framerate 1 -i algorithm_name_dir/step_%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p output_name.mp4
```

### Parameters:
- `algorithm_name_dir/step_%03d.png`: Path to the images (step files) generated during the algorithm execution. `%03d` will be replaced with the step number (e.g., `step_001.png`, `step_002.png`, etc.).
- `output_name.mp4`: The desired name of the output video.

Ensure you have **ffmpeg** installed on your system to use the above command. You can download it from [FFmpeg official website](https://ffmpeg.org/download.html).
