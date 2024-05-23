import os
from utils import *
from tqdm import tqdm
from initializer import *
from pyod.models.ecod import ECOD
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from torch_geometric.utils import degree
import numpy as np
from torch_scatter import scatter_mean

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

def filter_subgraph(args, data, anomaly_score):

    # obtain the candidate subgraph
    threshold = np.percentile(anomaly_score, q=args.q)
    candi_groups, residal_data, sub_size = GraphProcessor().sample_sub(data, anomaly_score, threshold)
    return residal_data, candi_groups, sub_size


def train_GAE(args, data, model, optimizer=None, is_training=True):
    x, edge_index, A = data.x.float(), data.edge_index, data.A
    if args.ours:
        if is_training:
            model.train()
            with tqdm(total=args.gcl_epochs, desc='(GAE)') as pbar:
                for epoch in range(1, args.gae_epochs):
                    norm = degree(data.edge_index[0], data.num_nodes)+degree(data.edge_index[1], data.num_nodes)
                    stru_recon, attr_recon = model(x, edge_index)
                    stru_score = ((torch.square(stru_recon - A).sum(1).sqrt())+(torch.square(stru_recon**2 - A**2).sum(1).sqrt())+(torch.square(stru_recon**3 - A**3).sum(1).sqrt()))
                    # stru_score = (stru_score - stru_score.min()) / (stru_score.max() - stru_score.min())
                    # stru_score = (torch.square(stru_recon - A).sum(1) + torch.square(stru_reconp - Ap).sum(1))/2
                    attr_score = torch.square(attr_recon - x).sum(1)
                    # attr_score = (attr_score - attr_score.min()) / (attr_score.max() - attr_score.min())
                    if args.alpha == 0:
                        score = attr_score
                    else:
                        score = args.alpha * stru_score + (1 - args.alpha) * attr_score
                    loss = score.mean()
                    total_error = score.clone().detach().cpu().numpy()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update()
                # for epoch in range(1, args.gae_epochs):
                #     stru_recon, attr_recon = model(x, edge_index)
                #     stru_score = torch.square(stru_recon**3 - A).sum(1)
                #     # stru_score = (torch.square(stru_recon - A).sum(1) + torch.square(stru_reconp - Ap).sum(1))/2
                #     attr_score = torch.square(attr_recon - x).sum(1)
                #     score = args.alpha * stru_score + (1 - args.alpha) * attr_score
                #     loss = score.mean()
                #     total_error = score.clone().detach().cpu().numpy()

                #     optimizer.zero_grad()
                #     loss.backward()
                #     optimizer.step()

                #     pbar.set_postfix({'loss': loss.item()})
                #     pbar.update()


        else:
            with torch.no_grad():
                A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
                norm = degree(data.edge_index[0], data.num_nodes)+degree(data.edge_index[1], data.num_nodes)
                stru_recon, attr_recon = model(data.x, data.edge_index)
                stru_score = ((torch.square(stru_recon - A).sum(1).sqrt())+(torch.square(stru_recon**2 - A**2).sum(1).sqrt())+(torch.square(stru_recon**3 - A**3).sum(1).sqrt()))
                attr_score = torch.square(attr_recon - data.x).sum(1).sqrt()
                # attr_score = (attr_score - attr_score.min()) / (attr_score.max() - attr_score.min())
                # stru_score = (stru_score - stru_score.min()) / (stru_score.max() - stru_score.min())

                if args.alpha == 0:
                    score = attr_score
                else:
                    score = args.alpha * stru_score + (1 - args.alpha) * attr_score
                total_error = score.detach().cpu().numpy()


    else:
        if is_training:
            model.train()
            with tqdm(total=args.gcl_epochs, desc='(GAE)') as pbar:
                for epoch in range(1, args.gae_epochs):
                    stru_recon, attr_recon = model(x, edge_index)
                    stru_score = torch.square(stru_recon ** 3 - A).sum(1)
                    # stru_score = (torch.square(stru_recon - A).sum(1) + torch.square(stru_reconp - Ap).sum(1))/2
                    attr_score = torch.square(attr_recon - x).sum(1)
                    score = args.alpha * stru_score + (1 - args.alpha) * attr_score
                    loss = score.mean()
                    total_error = score.clone().detach().cpu().numpy()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update()

        else:
            with torch.no_grad():
                A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
                stru_recon, attr_recon = model(data.x, data.edge_index)
                stru_score = torch.square(stru_recon ** 3 - A).sum(1).sqrt()
                attr_score = torch.square(attr_recon - data.x).sum(1).sqrt()
                score = args.alpha * stru_score + (1 - args.alpha) * attr_score
                total_error = score.detach().cpu().numpy()

    return total_error


def train_GCL(args, encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    optimizer.zero_grad()

    batch_list, cycle_edges_list, tree_root_list, path_middle_list, one_degree_list =\
    dataloader[0], dataloader[1], dataloader[2], dataloader[3], dataloader[4]

    for idx in range(len(batch_list)):
        data = batch_list[idx]
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, g0, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch, cycle_edges_list[idx],
                                           tree_root_list[idx], path_middle_list[idx], one_degree_list[idx])
        g0, g1, g2 = [encoder_model.encoder.project(g) for g in [g0, g1, g2]]
        # loss, _ = contrast_model(g1=g1, g2=g2, batch=data.batch)
        #- g0 1: 是通过原始图（未经增强）的节点特征x和边索引edge_index经过编码器处理后得到的全局图表示。
        # - g1: 是通过第一个图增强方法处理后的图得到的全局图表示。
        # - g2: 是通过第二个图增强方法处理后的图得到的全局图表示。
        # approximate mutual information via a model 
        # inner_epochs = args.inner_epochs
        # optimizer_local = torch.optim.Adam(contrast_model.parameters(), lr=args.inner_lr)
        # for j in range(0, inner_epochs):
        #     optimizer_local.zero_grad()

        #     shuffle_g0, shuffle_g1, shuffle_g2 = g0[torch.randperm(g0.shape[0])], g1[torch.randperm(g1.shape[0])], g2[torch.randperm(g2.shape[0])]
        #     joint1, joint2 = contrast_model(g1, g2), contrast_model(g0, g1)
        #     margin1, margin2 = contrast_model(g1, shuffle_g2), contrast_model(g0, shuffle_g1)
        #     mi = - (torch.mean(joint1) - torch.log(torch.mean(torch.exp(margin1)))) + \
        #          (torch.mean(joint2) - torch.log(torch.mean(torch.exp(margin2))))

        #     local_loss = mi
        #     local_loss.backward(retain_graph=True)
        #     optimizer_local.step()

        shuffle_g0, shuffle_g1, shuffle_g2 = g0[torch.randperm(g0.shape[0])], g1[torch.randperm(g1.shape[0])], g2[
            torch.randperm(g2.shape[0])]
        # joint1, joint2 = contrast_model(g1, g2), contrast_model(g0, g1)
        # margin1, margin2 = contrast_model(g1, shuffle_g2), contrast_model(g0, shuffle_g1)
        # mi = - (torch.mean(joint1) - torch.log(torch.mean(torch.exp(margin1)))) + \
            #  (torch.mean(joint2) - torch.log(torch.mean(torch.exp(margin2))))

        ###INFONCE
        loss = info_nce_loss(g0,g1,shuffle_g1)  
            
        # loss = mi
        # loss = F.relu(mi)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss, [encoder_model, contrast_model]


def train(args, data):

    # inizialize models
    GAE, GCL, opt_gae, opt_gcl = initialize_model(args)

    # train GAE to locate subgraph
    errors = train_GAE(args, data, GAE, optimizer=opt_gae, is_training=True)

    # filter
    residal_data, candi_groups, sub_size = filter_subgraph(args, data, errors)

    # preprocessing for locate critical edges and nodes of each batch
    batch_list, cycle_edges_list, tree_root_list, path_middle_list, one_degree_list = [], [], [], [], []
    for rdata in [residal_data]:
        cycle_edges, tree_root_nodes, del_edge_index, one_degree_nodes = GraphProcessor().pattern_search(rdata)
        batch_list.append(rdata)
        cycle_edges_list.append(cycle_edges)
        tree_root_list.append(tree_root_nodes)
        path_middle_list.append(del_edge_index)
        one_degree_list.append(one_degree_nodes)

    # train GCL
    with tqdm(total=args.gcl_epochs, desc='(GCL)') as pbar:
        for epoch in range(1, args.gcl_epochs):
            loss, GCL = train_GCL(args, GCL[0], GCL[1],
        [batch_list, cycle_edges_list, tree_root_list, path_middle_list, one_degree_list], opt_gcl)
            pbar.set_postfix({'loss': loss})
            pbar.update()

        return GAE, GCL, batch_list


def test(args, GAE, GCL, data):

    GCL_encoder, contrast_model = GCL[0], GCL[1]
    GAE.eval()
    GCL_encoder.eval()

    if args.benchmark:
        errors = np.load(f'bond_scores/{args.benchmarkmodel}_{args.benchmarkdataset}.npy')
        residal_data, candi_groups, sub_size = filter_subgraph(args, data, errors)
        g = scatter_mean(residal_data.x, residal_data.batch, dim=0)
       
    else:
        errors = train_GAE(args, data, GAE, optimizer=None, is_training=False)
        residal_data, candi_groups, sub_size = filter_subgraph(args, data, errors)

        batch_list, cycle_edges_list, tree_root_list, path_middle_list, one_degree_list = [], [], [], [], []
        for rdata in [residal_data]:
            cycle_edges, tree_root_nodes, del_edge_index, one_degree_nodes = GraphProcessor().pattern_search(rdata)
            batch_list.append(rdata)
            cycle_edges_list.append(cycle_edges)
            tree_root_list.append(tree_root_nodes)
            path_middle_list.append(del_edge_index)
            one_degree_list.append(one_degree_nodes)
        _, g, _, _, _, _ = GCL_encoder(residal_data.x, residal_data.edge_index, residal_data.batch)
    from torch_scatter import scatter_mean


    # g = scatter_mean(residal_data.x, residal_data.batch, dim=0)
    x, y = [], []
    x.append(g)
    y.append(residal_data.y)
    x = torch.cat(x, dim=0).detach().cpu().numpy()
    y = torch.cat(y, dim=0).detach().cpu().numpy()
    cls = ECOD(contamination=args.contamination).fit(x)

    y_score, y_pre = cls.decision_scores_, cls.labels_
    test_micro = f1_score(y, y_pre, average='micro')
    try:
        auc = roc_auc_score(y, y_score)
    except:
        auc = 0
    cr = CR_calculator(data, candi_groups, y_pre)
    node_auc = roc_auc_score(data.y[data.batch].cpu(), errors)
    node_ap = average_precision_score(data.y[data.batch].cpu(), errors)
    result = {'f1': test_micro, 'auc': auc, 'cr': cr, 'comp_size': sub_size, 'node_auc': node_auc, 'node_ap': node_ap}
    return result


def info_nce_loss(g0, g_pos, g_negs, temperature=0.1):
    """
    计算InfoNCE损失。
    :param g0: 原始图表示，形状为(batch_size, dim)
    :param g_pos: 正样本图表示，形状为(batch_size, dim)
    :param g_negs: 负样本图表示，形状为(batch_size, num_negs, dim)
    :param temperature: 温度参数，控制软化程度
    :return: InfoNCE损失
    """
    # 计算g0与g_pos之间的相似度
    pos_sim = torch.sum(g0 * g_pos, dim=-1) / temperature  # 形状为(batch_size,)
    
    # 计算g0与每个g_neg之间的相似度
    neg_sim = torch.sum(g0.unsqueeze(1) * g_negs, dim=-1) / temperature  # 形状为(batch_size, num_negs)
    
    # 计算损失
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # 形状为(batch_size, 1 + num_negs)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=g0.device)  # 正样本的标签为0
    loss = F.cross_entropy(logits, labels)
    
    return loss