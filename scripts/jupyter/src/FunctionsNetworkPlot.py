import glob
import os
import pandas as pd
import numpy as np
import gzip
from collections import defaultdict
import networkx as nx
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
mpl.rcParams['axes.linewidth'] = 1.4 #set the value globally

# Return 
# Edges (list of tuples with connections) and
# Positions (dictionary with tuple position 'node':(x,y,z)) 
def positions_GML(N, dim, alpha_a, alpha_g, filename):
    import gzip

    path = f"../../network/N_{N}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}/gml/"
    file = path + filename

    positions = {}
    edges = []
    
    with gzip.open(file, 'rt', encoding='utf-8') as f:
        current_node_id = None
        current_x = current_y = current_z = None
        in_graphics = False
        in_node = False
        in_edge = False
        current_source = current_target = None
        
        for line in f:
            line = line.strip()
            if line == "node":
                in_node = True
            elif line == "edge":
                in_edge = True
            elif line == "[":
                continue
            elif line == "]":
                if in_node and current_node_id is not None:
                    positions[f'id_{current_node_id}'] = (current_x, current_y, current_z)
                    current_node_id = current_x = current_y = current_z = None
                    in_node = False
                elif in_edge and current_source is not None:
                    edges.append((f'id_{current_source}', f'id_{current_target}'))
                    current_source = current_target = None
                    in_edge = False
                in_graphics = False
            elif in_node:
                if line.startswith("id "):
                    current_node_id = int(line.split()[1])
                elif line == "graphics":
                    in_graphics = True
                elif in_graphics:
                    if line.startswith("x "):
                        current_x = float(line.split()[1])
                    elif line.startswith("y "):
                        current_y = float(line.split()[1])
                    elif line.startswith("z "):
                        current_z = float(line.split()[1])
            elif in_edge:
                if line.startswith("source "):
                    current_source = int(line.split()[1])
                elif line.startswith("target "):
                    current_target = int(line.split()[1])
    
    return edges, positions


# Select one file .gml in specific folder (N, dim, alpha_a, alpha_g)
# Return:
# filename
def select_first_gml_gz_file(N, dim, alpha_a, alpha_g):
    directory = f"../../network/N_{N}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}/gml/"
    # Itera sobre os arquivos na pasta fornecida
    for file in os.listdir(directory):
        if file.endswith('.gml.gz'):
            selected_file = file
            #print(selected_file)
            return selected_file
    
    print("Nenhum arquivo .gml.gz encontrado na pasta.")
    return None


def get_limits_for_first_subplot(pos):
    """ Calcula limites para garantir que o primeiro subplot não seja cortado """
    x_values = np.array([x for x, y in pos.values()])
    y_values = np.array([y for x, y in pos.values()])

    x_min, x_max = x_values.min(), x_values.max()
    y_min, y_max = y_values.min(), y_values.max()

    # Adicionar margem para melhor visualização
    margin_x = (x_max - x_min) * 0.10  
    margin_y = (y_max - y_min) * 0.10

    return (x_min - margin_x, x_max + margin_x), (y_min - margin_y, y_max + margin_y)

def calculate_center_of_mass(pos):
    """Calcula o centro de massa do grafo"""
    x_cm = sum(x for x, y in pos.values()) / len(pos)
    y_cm = sum(y for x, y in pos.values()) / len(pos)
    return x_cm, y_cm

def draw_graph_subplots(ax, G, pos, title, row, col, adjust_limits=False):
    # Criar um degradê de cores usando um mapa de cores atualizado
    cmap = plt.get_cmap('viridis', len(pos))
    norm = mcolors.Normalize(vmin=0, vmax=len(pos) - 1)
    
    # Definir cores para os nós, variando de acordo com o índice
    node_colors = [cmap(norm(i)) for i in range(len(pos))]

    # Ordenar os nós numericamente
    node_list = sorted(pos.keys(), key=lambda x: int(x.split('_')[1]))

    # **Calcular centro de massa**
    x_cm, y_cm = calculate_center_of_mass(pos)
    G.add_node("cm", pos=(x_cm, y_cm))  # Adicionar o nó do centro de massa
    pos["cm"] = (x_cm, y_cm)  # Atualizar dicionário com a nova posição

    # Desenhar arestas primeiro
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#000000', alpha=0.6, width=0.8, style='solid')

    # Desenhar nós com cores do degradê
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_list, node_size=120,
                            node_color=node_colors, edgecolors='#000000', linewidths=0.8)

    # **Desenhar o centro de massa destacado**
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=["cm"], node_size=120, 
                            node_color="red", edgecolors="black", linewidths=2.0)

    ax.set_title(title, fontsize=35, pad=15)

    # Ajustar os limites apenas para o primeiro subplot
    if adjust_limits:
        global_x_limits, global_y_limits = get_limits_for_first_subplot(pos)
        ax.set_xlim(global_x_limits)
        ax.set_ylim(global_y_limits)

    # Ajustar margem para evitar corte dos ticks
    ax.margins(0.1)

    # Restaurando os ticks corretamente
    ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num=3))
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num=4))

    # Exibir os rótulos dos eixos conforme especificado
    if col == 0:
        ax.set_ylabel(r'$Y$', fontsize=35)
    else:
        ax.set_ylabel("")

    if row == 2:
        ax.set_xlabel(r'$X$', fontsize=35)
    else:
        ax.set_xlabel("")
    
    # **Forçar a exibição dos valores dos ticks**
    ax.tick_params(axis='both', which='both', length=4.0, width=2.0, direction='in', labelsize=25,
                   labelbottom=True, labelleft=True, pad=15.0)

    # **Garantir que os números dos ticks apareçam**
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_visible(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    # **Remover completamente as grades**
    ax.grid(False)

    
# draw one single network of dimensions dimension = 1, 2 or 3
def draw_graph(ax, G, pos, title, dim=2):
    """
    Função para desenhar um grafo em 1D, 2D ou 3D.
    
    Parâmetros:
    - ax: eixo do Matplotlib
    - G: grafo do NetworkX
    - pos: dicionário com posições dos nós
    - title: título do gráfico
    - dim: dimensão do espaço (1, 2 ou 3)
    """

    # Configurar título
    ax.set_title(title, fontsize=25)

    # Separar coordenadas dos nós
    x_values = np.array([p[0] for p in pos.values()])
    
    if dim >= 2:
        y_values = np.array([p[1] for p in pos.values()])
    
    if dim == 3:
        z_values = np.array([p[2] for p in pos.values()])

    # Ajustar os limites dos eixos
    x_min, x_max = x_values.min() - 1, x_values.max() + 1
    ax.set_xlim(x_min, x_max)

    if dim >= 2:
        y_min, y_max = y_values.min() - 1, y_values.max() + 1
        ax.set_ylim(y_min, y_max)

    if dim == 3:
        z_min, z_max = z_values.min() - 1, z_values.max() + 1
        ax.set_zlim(z_min, z_max)

    # Desenhar arestas
    for edge in G.edges():
        p1, p2 = pos[edge[0]], pos[edge[1]]
        if dim == 1:
            ax.plot([p1[0], p2[0]], [0, 0], color='#000000', alpha=0.8, lw=2)  # Linha horizontal
        elif dim == 2:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#000000', alpha=1.0)
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='#000000', alpha=1.0)

    # Destacar o nó na origem (0,0,0) ou (0,0) ou (0)
    special_node = None
    for node, coords in pos.items():
        if (dim == 1 and coords[0] == 0) or \
           (dim == 2 and coords == (0, 0)) or \
           (dim == 3 and coords == (0, 0, 0)):
            special_node = node

    # Desenhar nós normais
    if dim == 1:
        ax.scatter(x_values, np.zeros_like(x_values), s=600, c='#ff6666', edgecolors='black')
    elif dim == 2:
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=700, node_color='#ff6666', edgecolors='black', hide_ticks=False)
    else:
        ax.scatter(x_values, y_values, z_values, s=300, c='#ff6666', edgecolors='black')

    # Desenhar o nó da origem por último para sobrepor os demais
    if special_node:
        if dim == 1:
            ax.scatter([0], [0], s=900, c='#66b3ff', edgecolors='black', linewidths=2.5)
        elif dim == 2:
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[special_node], node_size=900, 
                                   node_color='#66b3ff', edgecolors='black', linewidths=2.5, hide_ticks=False)
        else:
            ax.scatter([0], [0], [0], s=900, c='#66b3ff', edgecolors='black', linewidths=2.5)


    # Ajustes específicos para dim=3
    if dim == 3:
        ax.set_box_aspect([1, 1, 1])  # Mantém a proporção igual entre os eixos
        ax.view_init(elev=20, azim=30)  # Ajusta a perspectiva para melhor visualização

        # Melhorar a visibilidade dos ticks e rótulos dos eixos
        ax.xaxis.labelpad = 15  # Move rótulo X para longe
        ax.yaxis.labelpad = 15  # Move rótulo Y para longe
        ax.zaxis.labelpad = 15  # Move rótulo Z para longe
        
        ax.tick_params(axis='x', pad=5)  # Ajusta espaçamento dos ticks no eixo X
        ax.tick_params(axis='y', pad=5)  # Ajusta espaçamento dos ticks no eixo Y
        ax.tick_params(axis='z', pad=5)  # Ajusta espaçamento dos ticks no eixo Z

        # Fechar a box corretamente no Matplotlib 3.4+
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True)  # Ativar a grid principal

    # Exibir rótulos dos eixos conforme a dimensão
    if dim != 3:
        ax.set_xlabel(r'$X$', fontsize=25)
        ax.set_ylabel(r'$Y$', fontsize=25)
    elif dim == 3:
        ax.set_xlabel(r'$X$', fontsize=25)
        ax.set_ylabel(r'$Y$', fontsize=25)
        ax.set_zlabel(r'$Z$', fontsize=25)

    # Garantir que os valores numéricos dos eixos apareçam
    ax.tick_params(axis='both', which='both', width=1.4, length=16.0, direction='in',labelsize=25)