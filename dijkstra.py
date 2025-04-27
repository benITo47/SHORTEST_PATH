import networkx as nx
import matplotlib.pyplot as plt
import heapq
import os

output_dir = "dijkstra_steps"
os.makedirs(output_dir, exist_ok=True)

G = nx.Graph()


edges = [
    (0, 1, 6),
    (0, 2, 5),
    (0, 3, 5),
    (1, 4, 8),
    (2, 1, 4),
    (2, 4, 1),
    (3, 2, 6),
    (3, 5, 1),
    (4, 6, 3),
    (5, 6, 3),
    (4, 7, 2),
    (5, 8, 4),
    (7, 6, 3),
    (8, 6, 2),
]
G.add_weighted_edges_from(edges)

pos = {
    0: (0, 0),
    1: (1, 1),
    2: (1, -1),
    3: (1, -3),
    4: (2, 1),
    5: (2, -2),
    6: (4, 0),
    7: (3, 2),
    8: (3, -2),
}


# Draw function
def draw_graph(G, pos, distances, visited, current=None, neighbor=None, step=0):
    plt.figure(figsize=(10, 8))
    edge_labels = nx.get_edge_attributes(G, "weight")

    node_colors = []
    for node in G.nodes():
        if node == current:
            node_colors.append("red")
        elif node == neighbor:
            node_colors.append("orange")
        elif node in visited:
            node_colors.append("lightgreen")
        else:
            node_colors.append("lightblue")

    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if (u == current and v == neighbor) or (v == current and u == neighbor):
            edge_colors.append("red")
            edge_widths.append(3)
        else:
            edge_colors.append("black")
            edge_widths.append(1)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        edge_color=edge_colors,
        width=edge_widths,
        node_size=700,
        font_size=10,
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    for node, (x, y) in pos.items():
        plt.text(
            x,
            y + 0.25,
            f"dist={distances.get(node, '∞')}",
            horizontalalignment="center",
            fontsize=9,
            color="black",
        )

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.title(f"Step {step}")
    plt.axis("off")
    plt.savefig(f"{output_dir}/step_{step:03d}.png")
    plt.close()


def dijkstra_visualized(G, start):
    distances = {node: float("inf") for node in G.nodes}
    distances[start] = 0
    visited = set()
    heap = [(0, start)]
    step = 0

    draw_graph(G, pos, distances, visited, step=step)

    while heap:
        current_distance, current_node = heapq.heappop(heap)

        if current_node in visited:
            continue

        visited.add(current_node)
        step += 1
        draw_graph(G, pos, distances, visited, current=current_node, step=step)

        for neighbor in G.neighbors(current_node):
            if neighbor not in visited:
                weight = G[current_node][neighbor]["weight"]
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(heap, (distance, neighbor))

                step += 1
                draw_graph(
                    G,
                    pos,
                    distances,
                    visited,
                    current=current_node,
                    neighbor=neighbor,
                    step=step,
                )

    step += 1
    draw_graph(G, pos, distances, visited, step=step)


dijkstra_visualized(G, start=4)
