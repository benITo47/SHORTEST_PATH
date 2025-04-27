import networkx as nx
import matplotlib.pyplot as plt
import os
import math

output_dir = "bellman_ford_steps"
os.makedirs(output_dir, exist_ok=True)

G = nx.DiGraph()


edges = [
    (0, 1, 6),
    (0, 2, 5),
    (0, 3, 5),
    (1, 4, -1),
    (2, 1, -2),
    (2, 4, 1),
    (3, 2, -2),
    (3, 5, -1),
    (4, 6, 3),
    (5, 6, 3),
    (4, 7, 2),
    (5, 8, 4),
    (7, 6, 1),
    (8, 6, -2),
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


def draw_graph_bf(
    G,
    pos,
    distances,
    visited,
    current=None,
    step=0,
    start=None,
    relaxed_edge=None,
    relaxed_node=None,
    iteration=1,
    finalized_edges=set(),
):
    plt.figure(figsize=(12, 10))
    edge_labels = nx.get_edge_attributes(G, "weight")

    node_colors = []
    for node in G.nodes():
        if node == current:
            node_colors.append("red")
        elif node == start:
            node_colors.append("blue")
        elif node == relaxed_node:
            node_colors.append("yellow")
        elif node in visited:
            node_colors.append("lightgreen")
        else:
            node_colors.append("lightblue")

    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if relaxed_edge and (u, v) == relaxed_edge:
            edge_colors.append("red")
            edge_widths.append(3)
        elif (u, v) in finalized_edges:
            edge_colors.append("green")
            edge_widths.append(2)
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
            y + 0.1,
            f"dist={distances.get(node, '∞')}",
            horizontalalignment="center",
            fontsize=9,
            color="black",
        )

    plt.text(
        0.02,
        0.98,
        f"Iteration: {iteration}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="left",
        color="black",
        weight="bold",
    )

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.axis("off")
    plt.savefig(f"{output_dir}/step_{step:03d}.png")
    plt.close()


def bellman_ford(G, start):
    distances = {node: float("inf") for node in G.nodes}
    predecessors = {node: None for node in G.nodes}
    distances[start] = 0
    visited = set()
    step = 0
    relaxed_edges = set()

    iteration = 1
    draw_graph_bf(
        G,
        pos,
        distances,
        visited,
        current=start,
        step=step,
        start=start,
        iteration=iteration,
    )
    for _ in range(len(G.nodes) - 1):
        step += 1
        draw_graph_bf(
            G,
            pos,
            distances,
            visited,
            current=start,
            step=step,
            start=start,
            iteration=iteration,
        )
        for u, v, weight in G.edges(data=True):
            if distances[u] + weight["weight"] < distances[v]:

                step += 1
                draw_graph_bf(
                    G,
                    pos,
                    distances,
                    visited,
                    current=u,
                    step=step,
                    start=start,
                    relaxed_node=u,
                    iteration=iteration,
                )
                distances[v] = distances[u] + weight["weight"]
                predecessors[v] = u
                step += 1
                draw_graph_bf(
                    G,
                    pos,
                    distances,
                    visited,
                    current=u,
                    step=step,
                    start=start,
                    relaxed_node=v,
                    relaxed_edge=(u, v),
                    iteration=iteration,
                )

                relaxed_edges.add((u, v))

        iteration += 1
        relaxed_edges.clear()

    iteration -= 1
    draw_graph_bf(
        G,
        pos,
        distances,
        visited,
        current=start,
        step=step,
        start=start,
        iteration=iteration,
    )
    return distances, predecessors


distances, predecessors = bellman_ford(G, start=0)
print("Final Distances:", distances)
print("Paths:", predecessors)
