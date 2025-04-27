import networkx as nx
import matplotlib.pyplot as plt
import heapq
import os
import math

output_dir = "a_star_steps"
os.makedirs(output_dir, exist_ok=True)

G = nx.Graph()

edges = [
    (0, 1, 4),
    (0, 2, 3),
    (1, 2, 1),
    (2, 3, 8),
    (2, 4, 7),
    (3, 4, 2),
    (4, 5, 6),
    (5, 6, 9),
    (6, 7, 13),
    (7, 8, 4),
    (8, 9, 2),
    (5, 9, 6),
    (3, 6, 5),
    (1, 5, 7),
    (6, 4, 1),
]
G.add_weighted_edges_from(edges)

pos = {
    0: (0, 0),
    1: (2, 0),
    2: (2, 1),
    3: (3, 1),
    4: (4, 2),
    5: (6, 1),
    6: (5, 0),
    7: (8, 1),
    8: (9, 2),
    9: (8, 2),
}


def heuristic(a, b):
    x1, y1 = pos[a]
    x2, y2 = pos[b]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def draw_graph(
    G,
    pos,
    g_score,
    f_score,
    visited,
    current=None,
    neighbor=None,
    step=0,
    start=None,
    goal=None,
    path=None,
):
    plt.figure(figsize=(12, 10))
    edge_labels = nx.get_edge_attributes(G, "weight")

    node_colors = []
    for node in G.nodes():
        if node == current:
            node_colors.append("red")
        elif node == neighbor:
            node_colors.append("orange")
        elif node == start:
            node_colors.append("blue")
        elif node == goal:
            node_colors.append("green")
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
            y + 0.1,
            f"f={f'{f_score[node]:.2f}' if node in f_score else '∞'}",
            horizontalalignment="center",
            fontsize=9,
            color="black",
        )

    if path:
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(
            G, pos, edgelist=path_edges, edge_color="blue", width=3, alpha=0.6
        )

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.title(f"Step {step}")
    plt.axis("off")
    plt.savefig(f"{output_dir}/step_{step:03d}.png")
    plt.close()


def reconstruct_path(came_from, current_node):
    path = [current_node]
    while current_node in came_from:
        current_node = came_from[current_node]
        path.append(current_node)
    path.reverse()
    return path


def a_star_visualized(G, start, goal):
    closedset = set()
    openset = {start}
    g_score = {node: float("inf") for node in G.nodes}
    g_score[start] = 0
    h_score = {node: heuristic(node, goal) for node in G.nodes}
    f_score = {node: float("inf") for node in G.nodes}
    f_score[start] = h_score[start]

    came_from = {}
    step = 0

    draw_graph(
        G,
        pos,
        g_score,
        f_score,
        closedset,
        step=step,
        start=start,
        goal=goal,
    )

    while openset:
        current_node = min(openset, key=lambda node: f_score[node])

        if current_node == goal:
            path = reconstruct_path(came_from, goal)
            print(f"Goal {goal} reached!")
            step += 1
            draw_graph(
                G,
                pos,
                g_score,
                f_score,
                closedset,
                step=step,
                start=start,
                goal=goal,
                path=path,
            )
            return path

        openset.remove(current_node)
        closedset.add(current_node)
        step += 1
        draw_graph(
            G,
            pos,
            g_score,
            f_score,
            closedset,
            current=current_node,
            step=step,
            start=start,
            goal=goal,
        )

        for neighbor in G.neighbors(current_node):
            if neighbor in closedset:
                continue

            tentative_g_score = (
                g_score[current_node] + G[current_node][neighbor]["weight"]
            )

            tentative_is_better = False
            if neighbor not in openset:
                openset.add(neighbor)
                tentative_is_better = True
            elif tentative_g_score < g_score[neighbor]:
                tentative_is_better = True

            if tentative_is_better:
                came_from[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + h_score[neighbor]

                step += 1
                draw_graph(
                    G,
                    pos,
                    g_score,
                    f_score,
                    closedset,
                    current=current_node,
                    neighbor=neighbor,
                    step=step,
                    start=start,
                    goal=goal,
                )

    print("No path found!")
    return None


a_star_visualized(G, start=0, goal=9)
