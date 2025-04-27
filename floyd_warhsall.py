import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.patches as mpatches

output_dir = "floyd_warshall_steps"
os.makedirs(output_dir, exist_ok=True)

G = nx.DiGraph()

# Najpierw jawnie dodajemy węzły w zadanej kolejności
G.add_nodes_from([0, 1, 2, 3])

# Dodajemy krawędzie
edges = [
    (0, 2, -2),
    (1, 0, 4),
    (1, 2, 3),
    (2, 3, 2),
    (3, 1, -1),
]
G.add_weighted_edges_from(edges)

# Pozycje węzłów na rysunku
pos = {
    0: (1, 2),
    1: (0, 1),
    2: (2, 1),
    3: (1, 0),
}


def draw_graph_fw(
    G,
    pos,
    distances,
    visited,
    step=0,
    iteration=1,
    updated_edges=None,
    distance_matrix=None,
    current_ijk=None,
    shadowed_cells=set(),
    next_node=None,
    final=False,
    highlight_edges_for_update=None,  # <-- DODANE
):
    fig, axs = plt.subplots(1, 2, figsize=(14, 8))

    ax_graph = axs[0]
    plt.sca(ax_graph)

    edge_labels = nx.get_edge_attributes(G, "weight")

    node_colors = []
    if current_ijk:
        i, k, j = current_ijk
    else:
        i = k = j = None

    nodes = sorted(G.nodes())
    for node in nodes:
        if node == i:
            node_colors.append("#00CC00")
        elif node == k:
            node_colors.append("#FFA500")
        elif node == j:
            node_colors.append("#FF0000")
        else:
            node_colors.append("lightblue")

    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if updated_edges and ((u, v) in updated_edges or (v, u) in updated_edges):
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
        font_size=12,
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    ax_table = axs[1]
    plt.sca(ax_table)

    matrix = np.array(distance_matrix, dtype=object)
    matrix = np.where(matrix == float("inf"), "∞", matrix)

    size = len(matrix)
    full_matrix = np.empty((size + 1, size + 1), dtype=object)
    full_matrix[0, 0] = "i/j"
    full_matrix[0, 1:] = [str(x) for x in nodes]
    full_matrix[1:, 0] = [str(x) for x in nodes]
    full_matrix[1:, 1:] = matrix

    cell_colours = [["white" for _ in range(size + 1)] for _ in range(size + 1)]
    for idx in range(size + 1):
        cell_colours[0][idx] = "lightgray"
        cell_colours[idx][0] = "lightgray"

    # Najpierw zaznacz pomocnicze (i,k) i (k,j)
    if highlight_edges_for_update:
        for ci, cj in highlight_edges_for_update:
            if ci in nodes and cj in nodes:
                row = nodes.index(ci) + 1
                col = nodes.index(cj) + 1
                cell_colours[row][col] = "#ADD8E6"  # jasny niebieski

    # Potem zaznacz główną aktualizowaną (i,j)
    for ci, cj in shadowed_cells:
        if ci in nodes and cj in nodes:
            row = nodes.index(ci) + 1
            col = nodes.index(cj) + 1
            cell_colours[row][col] = "lightgray"

    table = plt.table(
        cellText=full_matrix,
        loc="center",
        cellLoc="center",
        cellColours=cell_colours,
        bbox=[0, 0.2, 1, 0.7],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    if current_ijk:
        if i in nodes and j in nodes:
            hi = nodes.index(i) + 1
            hj = nodes.index(j) + 1
            if (hi, hj) in table.get_celld():
                cell = table[(hi, hj)]
                if (i, j) in shadowed_cells:
                    cell.set_facecolor("yellow")
                    cell.get_text().set_color("red")
                    cell.get_text().set_weight("bold")
                else:
                    cell.set_facecolor("#FFFFE0")
                    cell.get_text().set_color("black")
                    cell.get_text().set_weight("normal")

    ax_table.axis("off")

    if current_ijk:
        fig.suptitle(
            f"Step {step}: Checking i={i}, k={k}, j={j}",
            fontsize=13,
        )
    else:
        fig.suptitle(f"Step {step}", fontsize=14)

    if final and next_node:
        lines = []
        for i in nodes:
            for j in nodes:
                if i != j and distances[i][j] != float("inf"):
                    path = [i]
                    current = i
                    while current != j:
                        current = next_node[current][j]
                        if current is None:
                            break
                        path.append(current)
                    if path[-1] == j:
                        lines.append(
                            f"{i} → {j}: {' → '.join(map(str, path))} (cost {distances[i][j]})"
                        )
        text = "\n".join(lines)
        ax_table.text(
            0,
            -0.15,
            f"Shortest paths:\n{text}",
            transform=ax_table.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="left",
            wrap=True,
        )

    green_patch = mpatches.Patch(color="#00CC00", label="Start (i)")
    orange_patch = mpatches.Patch(color="#FFA500", label="Through (k)")
    red_patch = mpatches.Patch(color="#FF0000", label="End (j)")
    fig.legend(
        handles=[green_patch, orange_patch, red_patch],
        loc="lower center",
        ncol=3,
        fontsize=10,
        frameon=False,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"{output_dir}/step_{step:03d}.png")
    plt.close()


def floyd_warshall(G):
    nodes = sorted(G.nodes())
    dist = {node: {other_node: float("inf") for other_node in nodes} for node in nodes}
    next_node = {node: {other_node: None for other_node in nodes} for node in nodes}

    step = 0
    iteration = 0

    for i in nodes:
        dist[i][i] = 0
        distance_matrix = [[dist[ii][jj] for jj in nodes] for ii in nodes]
        draw_graph_fw(
            G,
            pos,
            dist,
            visited={i},
            step=step,
            iteration=iteration,
            updated_edges=None,
            distance_matrix=distance_matrix,
            current_ijk=(i, i, i),
            shadowed_cells={(i, i)},
        )
        step += 1

    for u, v, weight in G.edges(data=True):
        dist[u][v] = weight["weight"]
        next_node[u][v] = v
        distance_matrix = [[dist[ii][jj] for jj in nodes] for ii in nodes]
        draw_graph_fw(
            G,
            pos,
            dist,
            visited={u, v},
            step=step,
            iteration=iteration,
            updated_edges=[(u, v)],
            distance_matrix=distance_matrix,
            current_ijk=(u, u, v),
            shadowed_cells={(u, v)},
        )
        step += 1

    for k in nodes:
        for i in nodes:
            for j in nodes:
                updated = False
                used_edges = []
                if dist[i][k] != float("inf") and dist[k][j] != float("inf"):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_node[i][j] = next_node[i][k]
                        updated = True
                        used_edges = [(i, k), (k, j)]

                distance_matrix = [[dist[ii][jj] for jj in nodes] for ii in nodes]
                draw_graph_fw(
                    G,
                    pos,
                    dist,
                    {i, j},
                    step=step,
                    iteration=iteration,
                    updated_edges=used_edges if updated else None,
                    distance_matrix=distance_matrix,
                    current_ijk=(i, k, j),
                    shadowed_cells={(i, j)} if updated else set(),
                    highlight_edges_for_update=[(i, k), (k, j)] if updated else None,
                )
                step += 1

        iteration += 1

    distance_matrix = [[dist[ii][jj] for jj in nodes] for ii in nodes]
    draw_graph_fw(
        G,
        pos,
        dist,
        set(),
        step=step,
        iteration=iteration,
        distance_matrix=distance_matrix,
        next_node=next_node,
        final=True,
    )

    return dist, next_node


distances, next_node = floyd_warshall(G)
print("Shortest distances:", distances)
