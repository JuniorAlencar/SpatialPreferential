import glob
import os
import pandas as pd
import numpy as np
import gzip
from collections import defaultdict
import networkx as nx
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
mpl.rcParams['axes.linewidth'] = 1.4 #set the value globally

# Return 
# Edges (list of tuples with connections) and
# Positions (dictionary with tuple position 'node':(x,y,z)) 
def positions_GML(N, dim, alpha_a, alpha_g, filename):
    path = f"../../network/N_{N}/dim_{dim}/alpha_a_{alpha_a:.2f}_alpha_g_{alpha_g:.2f}/gml/"
    file = path + filename
    
    x,y,z = [],[],[]
    source = []
    target = []
    
    index = 0
    with open(file, 'rb') as file_in:
        lines = []
        # decompress gzip
        with gzip.GzipFile(fileobj=file_in, mode='rb') as gzip_file:
            for line in gzip_file:
                # decode file
                decoded_line = line.decode('utf-8')
                
                # positions nodes
                if(decoded_line[0]=="x"):
                    x.append(float(line[1:-1]))
                elif(decoded_line[0]=="y"):
                    y.append(float(line[1:-1]))                    
                elif(decoded_line[0]=="z"):
                    z.append(float(line[1:-1])) 
                              
                
                # edges nodes

                # Append to buffer
                lines.append(decoded_line) 
                
                if len(lines) >= 5:  # 1 (current line) + 2 (2 lines below) + 3 (3 lines below)
                    # Check if the line 3 lines before the current line starts with 'edge'
                    if lines[-4].startswith('edge'):
                        # Lines 2 and 3 below the 'edge' line
                        line_2_below = lines[-2]
                        line_3_below = lines[-1]
                        source.append('id_' + line_2_below.strip()[8:-1])
                        target.append('id_' + line_3_below.strip()[8:-1])
                        
    
    
    positions = {}
    for i in range(len(x)):
        positions[f'id_{i}'] = (x[i],y[i],z[i])
    edges = [(i,j) for i,j in zip(source,target)]
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

# draw network in suplots 2D
def draw_graph_subplots(ax, G, pos, title, row, col):
    # Definir cores para os nós, destacando o nó na origem
    node_colors = ['#ff6666' for _ in pos]  # Todos os nós inicialmente vermelhos
    special_node = None  # Para armazenar o nó na origem

    # Identificar se existe um nó na posição (0,0)
    for node, (x, y) in pos.items():
        if x == 0 and y == 0:
            special_node = node  # Guardar o nó da origem
            node_colors[list(pos.keys()).index(node)] = '#66b3ff'  # Azul-claro para a origem

    # Desenhar arestas primeiro
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#000000', hide_ticks=False)

    # Desenhar todos os nós (exceto o nó na origem, que será desenhado depois)
    node_list = list(pos.keys())
    if special_node:
        node_list.remove(special_node)  # Remove o nó da origem para desenhá-lo separadamente

    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_list, node_size=500,
                            node_color='#ff6666', edgecolors='#000000', hide_ticks=False)

    # Desenhar o nó da origem por último, para sobrepor os demais
    if special_node:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[special_node], node_size=700, 
                               node_color='#66b3ff', edgecolors='#000000', linewidths=2.5, hide_ticks=False)

    ax.set_title(title, fontsize=30)

    # Ajuste os limites dos eixos
    x_values = np.array([x for x, y in pos.values()])
    y_values = np.array([y for x, y in pos.values()])

    x_min, x_max = x_values.min() - 1, x_values.max() + 1
    y_min, y_max = y_values.min() - 1, y_values.max() + 1

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Definir os ticks para garantir que os números apareçam nos eixos
    ax.set_xticks(np.linspace(x_min, x_max, num=5))  # Ticks do eixo X
    ax.set_yticks(np.linspace(y_min, y_max, num=5))  # Ticks do eixo Y

    # Exibir os rótulos dos eixos conforme especificado
    if col == 0:
        ax.set_ylabel(r'$Y$', fontsize=17)  # Exibir rótulo do eixo Y na coluna da esquerda
    else:
        ax.set_ylabel("")  # Esconder rótulo do eixo Y

    if row == 2:
        ax.set_xlabel(r'$X$', fontsize=17)  # Exibir rótulo do eixo X na última linha
    else:
        ax.set_xlabel("")  # Esconder rótulo do eixo X
    
    # Garantir que os valores numéricos dos eixos apareçam em todos os plots
    ax.tick_params(axis='both', which='both', length=4.0, width=1.4, direction='in', labelsize=15)

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