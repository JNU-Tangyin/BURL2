import pickle
import networkx as nx
import matplotlib.pyplot as plt
from networkx.classes.reportviews import NodeView, NodeDataView
import plotly.graph_objects as go
import plotly.express as px  # 用于快速创建统计图表
from plotly.subplots import make_subplots
import dash
# import dash_core_components as dcc
from dash import dcc
# import dash_html_components as html
from dash import html
from dash.dependencies import Input, Output, State
import pandas as pd
from dash import dash_table
from tqdm import tqdm


def create_node_link_trace(G, pos):
    """
    功能：创建节点和边的3D轨迹
    实现：使用networkx的布局信息创建Plotly的Scatter3d对象
    """
    edge_x = []
    edge_y = []
    edge_z = []
    text_edge = []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
        text_edge.append(G[edge[0]][edge[1]]['name'])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=0.5, color='#888'),
        text=text_edge,
        hoverinfo='text',
        mode='lines')

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_z = [pos[node][2] for node in G.nodes()]
    # text_node = [G.nodes[i]['name'] for i in G.nodes()]

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in G.adjacency():
        node_adjacencies.append(len(adjacencies))
        node_text.append(f'Node: {node}<br># of connections: {len(adjacencies)} Node_class: {G.nodes[node]["name"]}')

    node_trace.marker.color = node_adjacencies  # 根据邻居数调整颜色深度
    node_trace.text = node_text

    return edge_trace, node_trace


def create_edge_label_trace(G, pos, edge_labels):
    """
    功能：创建边标签的3D轨迹
    实现：计算边的中点位置，创建Scatter3d对象显示标签
    """
    return go.Scatter3d(
        x=[pos[edge[0]][0] + (pos[edge[1]][0] - pos[edge[0]][0]) / 2 for edge in edge_labels],
        y=[pos[edge[0]][1] + (pos[edge[1]][1] - pos[edge[0]][1]) / 2 for edge in edge_labels],
        z=[pos[edge[0]][2] + (pos[edge[1]][2] - pos[edge[0]][2]) / 2 for edge in edge_labels],
        mode='text',
        text=list(edge_labels.values()),
        textposition='middle center',
        hoverinfo='none'
    )


def create_degree_distribution(G):
    """
    功能：创建节点度分布直方图
    实现：使用plotly.express创建直方图
    """
    degrees = [d for n, d in G.degree()]
    fig = px.histogram(x=degrees, nbins=20, labels={'x': 'Degree', 'y': 'Count'})
    fig.update_layout(
        title_text='Node Degree Distribution',
        margin=dict(l=0, r=0, t=30, b=0),
        height=300
    )
    return fig


def create_centrality_plot(G):
    """
    功能：创建节点中心性分布箱线图
    实现：计算度中心性，使用plotly.express创建箱线图
    """
    centrality = nx.degree_centrality(G)
    centrality_values = list(centrality.values())
    fig = px.box(y=centrality_values, labels={'y': 'Centrality'})
    fig.update_layout(
        title_text='Degree Centrality Distribution',
        margin=dict(l=0, r=0, t=30, b=0),
        height=300
    )
    return fig


def visualize_graph_plotly(G):
    """功能：使用Plotly创建全面优化布局的高级交互式知识图谱可视化
    实现：
        创建3D布局
        生成节点和边的轨迹
        创建子图，包括3D图、度分布图和中心性分布图
        添加交互式按钮和滑块
        优化整体布局
    """
    if G.number_of_nodes() == 0:
        print("Graph is empty. Nothing to visualize.")
        return

    pos = nx.spring_layout(G, dim=3)  # 3D layout
    edge_trace, node_trace = create_node_link_trace(G, pos)

    edge_labels = nx.get_edge_attributes(G, 'relation')
    edge_label_trace = create_edge_label_trace(G, pos, edge_labels)

    degree_dist_fig = create_degree_distribution(G)
    centrality_fig = create_centrality_plot(G)

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.7, 0.3],
        row_heights=[0.7, 0.3],
        specs=[
            [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
            [None, {"type": "xy"}]
        ],
        subplot_titles=(
            "3D Knowledge Graph Code by AI超元域频道", "Node Degree Distribution", "Degree Centrality Distribution")
    )

    fig.add_trace(edge_trace, row=1, col=1)
    fig.add_trace(node_trace, row=1, col=1)
    fig.add_trace(edge_label_trace, row=1, col=1)

    fig.add_trace(degree_dist_fig.data[0], row=1, col=2)
    fig.add_trace(centrality_fig.data[0], row=2, col=2)

    # Update 3D layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            zaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            aspectmode='cube'
        ),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    )

    # Add buttons for different layouts
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(args=[{"visible": [True, True, True, True, True]}], label="Show All", method="update"),
                    dict(args=[{"visible": [True, True, False, True, True]}], label="Hide Edge Labels",
                         method="update"),
                    dict(args=[{"visible": [False, True, False, True, True]}], label="Nodes Only", method="update")
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.05,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    # Add slider for node size
    fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Node Size: "},
            pad={"t": 50},
            steps=[dict(method='update',
                        args=[{'marker.size': [i] * len(G.nodes)}],
                        label=str(i)) for i in range(5, 21, 5)]
        )]
    )

    # 优化整体布局
    # fig.update_layout(
    #     height=1198,  # 增加整体高度
    #     width=2055,  # 增加整体宽度
    #     title_text="Advanced Interactive Knowledge Graph",
    #     margin=dict(l=10, r=10, t=25, b=10),
    #     legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    # )

    return fig


def build_graph():
    tlds = [
        'net', 'com', 'cn', 'top', 'mobi', 'tv', 'link', 'cloud', 'im',
        'cc', 'ms', 'me', 'biz', 'xyz', 'vn', 'io', 'fun', 'is',
        'org', 'fm', 'co', 'hk', 'info', 'vip', 'ink', 'xin',
        'club', 'us'
    ]
    try:
        df = pd.read_csv("data/host_split.csv")
    except:
        df = pd.read_csv("data/host_split.csv", encoding='gbk')
    df = df.sort_values(by="len")
    df = df.reset_index(drop=True)
    node = []
    edge = []
    for index in tqdm(df.index):
        node.append({"node_id": index, "node_class": df.loc[index, '标签']})
        for col in range(int(df.loc[index, 'len'])):
            tag = df.loc[index, str(col)]
            for iter_i in range(index, df.__len__()):
                goal = df.loc[iter_i, str(col)]
                if goal == tag and tag not in tlds:
                    edge.append({'edge': (index, iter_i), 'edge_name': tag})
                    # print("标签：{}, 主机地址：{}, 边：{}, 边标签：{}, 边主机地址: {}".format(
                    #     df.loc[index, '标签'],
                    #     df.loc[index, "主机地址"],
                    #     tag,
                    #     df.loc[iter_i, '标签'],
                    #     df.loc[iter_i, "主机地址"]
                    # ))
    with open("graph_info.thg", "wb") as w:
        pickle.dump((node, edge), w)


app = dash.Dash(__name__)

try:
    with open("graph_info.thg", "rb") as r:
        data = pickle.load(r)
except:
    print("graph_info.thg not found, begin build graph")
    build_graph()
    with open("graph_info.thg", "rb") as r:
        data = pickle.load(r)

G = nx.Graph()
node_list, edge_list = data
for i in node_list:
    G.add_node(i['node_id'], name=i['node_class'])
for i in edge_list:
    G.add_edge(i['edge'][0], i['edge'][1], name=i['edge_name'])

fig = visualize_graph_plotly(G)

app.layout = html.Div([
    dcc.Graph(
        id='example-graph',
        figure=fig,
        style={'width': '100%', "height": '900px'}
    ),
    dash_table.DataTable(id='click-data-table',
                         columns=[{"name": 'neighbor', "id": 'neighbor'}, {"name": 'edge', "id": 'edge'}],
                         data=[],
                         style_table={'width': '50%'})
])


@app.callback(
    Output('click-data-table', 'data'),
    [Input('example-graph', 'clickData')])
def display_click_data(clickData):
    print(clickData)
    if clickData and 'points' in clickData and len(clickData['points']) > 0 and 'text' in clickData['points'][0]:
        node = clickData['points'][0]['pointNumber']
        neighbors = list(G.neighbors(node))
        edges = [(node, neighbor) for neighbor in neighbors]
        df = pd.DataFrame(columns=['neighbor', 'edge'], index=range(neighbors.__len__()))
        for i in df.index:
            df.loc[i, 'neighbor'] = "{}:{}".format(neighbors[i], G.nodes[neighbors[i]]['name'])
            df.loc[i, 'edge'] = G[edges[i][0]][edges[i][1]]['name']
        # return f"Node: {node} Neighbors: {neighbors}<br>Edges: {edges}"
        print(df.to_dict('records'))
        return df.to_dict('records')
    else:
        res = [{'neighbor': '点击节点获取信息', 'edge': '点击节点获取信息'}]
        return res


if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
