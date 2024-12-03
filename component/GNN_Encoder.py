import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def add_self_loops(edge_index, num_nodes):
    """
    添加自环边（每个节点都和自己有一条边连接）
    """
    loop_index = torch.arange(num_nodes, dtype=torch.long).view(1, -1).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index


def gnn_extract_excel_features(filename):
    # 读取Excel数据
    readbook = pd.read_excel(f'{filename}', engine='openpyxl')

    # 患者 ID 和标签
    index = readbook.iloc[:, 0].to_numpy()  # 患者ID
    labels = readbook.iloc[:, -1].to_numpy()  # 标签（如HER2+/-等）

    # 临床病理特征（去掉患者ID和标签）
    features_df = readbook.iloc[:, 1:-1]

    # 打印列名和特征数量
    print(f"Feature columns: {features_df.columns}")
    print(f"Number of features: {features_df.shape[1]}")

    # 分析数值特征和类别特征
    numeric_features = features_df.select_dtypes(include=[np.number])
    categorical_features = features_df.select_dtypes(exclude=[np.number])

    # 输出数值特征和类别特征的数量
    print(f"Number of numeric features: {numeric_features.shape[1]}")
    print(f"Number of categorical features: {categorical_features.shape[1]}")

    # 对类别特征进行独热编码
    if not categorical_features.empty:
        categorical_features = pd.get_dummies(categorical_features)
        print(f"Number of categorical features after one-hot encoding: {categorical_features.shape[1]}")

    # 合并数值特征和类别特征
    combined_features = pd.concat([numeric_features, categorical_features], axis=1)
    combined_features = combined_features.to_numpy(dtype=np.float32)
    # 打印最终合并特征的数量
    print(
        f"Total number of features after combining numeric and one-hot encoded categorical features: {combined_features.shape[1]}")

    # 构建图数据：每个患者为一个图，每个临床特征为一个节点
    def create_graph_from_features(features):
        num_patients = features.shape[0]  # 患者数量
        num_features = features.shape[1]  # 每个患者的临床特征数量（每列一个特征）

        edge_index_list = []  # 存储所有患者的边信息
        node_features_list = []  # 存储所有患者的节点特征
        batch_list = []  # 存储患者的批处理信息

        for patient_idx in range(num_patients):
            patient_features = features[patient_idx]  # 当前患者的特征
            node_features_list.append(patient_features)

            # 构建边：基于特征间的关系
            edge_index = []
            for i in range(num_features):
                for j in range(num_features):
                    if i != j:  # 不创建自环，稍后会统一添加
                        edge_index.append([i, j])

            # 转换为 PyTorch 张量并添加自环
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_index = add_self_loops(edge_index, num_features)
            edge_index_list.append(edge_index)

            # 为每个患者设置一个独立的批处理 ID
            batch_list.extend([patient_idx] * num_features)

        # 合并所有患者的节点特征和边
        all_node_features = torch.tensor(np.concatenate(node_features_list, axis=0), dtype=torch.float)
        all_node_features = all_node_features.view(-1, num_features)  # 确保是二维张量 (num_nodes, num_features)

        all_edge_indices = torch.cat(edge_index_list, dim=1)
        batch_tensor = torch.tensor(batch_list, dtype=torch.long)  # 批处理信息

        # 使用 Batch 来将图数据合并，保持不同患者的独立性
        graph_data = Batch(x=all_node_features, edge_index=all_edge_indices, batch=batch_tensor)

        return graph_data

    # 创建图数据
    graph_data = create_graph_from_features(combined_features)

    # 检查 x 的维度
    print(f"x shape: {graph_data.x.shape}")
    print(f"edge_index shape: {graph_data.edge_index.shape}")

    # 定义图神经网络模型
    class GNN(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super(GNN, self).__init__()
            self.conv1 = GCNConv(in_channels, 128)
            self.conv2 = GCNConv(128, 64)
            self.conv3 = GCNConv(64, out_channels)

        def forward(self, x, edge_index, batch):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.conv3(x, edge_index)
            return x

    # 初始化模型
    model = GNN(in_channels=combined_features.shape[1], out_channels=combined_features.shape[1]).to(device)
    data = graph_data.to(device)

    # 使用图神经网络进行节点特征聚合
    model.eval()
    with torch.no_grad():
        aggregated_features = model(data.x, data.edge_index, data.batch).cpu().numpy()

    return index, aggregated_features, labels

#
# def main():
#     index, gnn_excel_feature, labels = gnn_extract_excel_features(
#         '/tmp/pycharm_project_357/HER2_excel_data/HER2-data.xlsx')
#     print(f"Patient IDs: {index[:10]}")
#     print(f"Aggregated Features: {gnn_excel_feature[:10]}")
#     print(f"Labels: {labels[:10]}")
#
#
# if __name__ == "__main__":
#     main()
