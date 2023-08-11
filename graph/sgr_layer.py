""""
Define a generic GRM layer model
"""
from pickletools import decimalnl_short
import  torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .graph_util import *
#from graph.coco_data import *
from omegaconf import OmegaConf
#from .global_settings import GPU_ID
from torch.autograd import Variable
from torch.nn import Parameter
import math
#from .init_weights import init_weights

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d

#cuda_suffix = 'cuda:' + str(GPU_ID) if len(str(GPU_ID)) == 1 else "cuda"
#device = torch.device(cuda_suffix if torch.cuda.is_available() else "cpu")




class LocalToSemantic(nn.Module):
    # [?, Dl, H, W]
    def __init__(self, input_feature_channels, visual_feature_channels,
                 num_symbol_node
                 ):
        super(LocalToSemantic, self).__init__()

        self.conv1 = nn.Conv2d(input_feature_channels, num_symbol_node,
                              kernel_size=1, stride=1)
        

        # 利用一层卷积将输入图片的channel由数目Dl转化为Dc, 即[？，Dc, H, W]
        self.conv2 = nn.Conv2d(input_feature_channels, 300,
                              kernel_size=1, stride=1)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        votes = self.conv1(x)
        votes = F.softmax(votes, dim=1)
        #[？，M, H, W]->[？，M, H*W]
        votes = votes.view(votes.size(0), votes.size(1), -1)
        #[？，Dc, H, W]
        in_feat = self.conv2(x)
        # [？，Dc, H, W]->[？，Dc, H*W]->[？, H*W , Dc]
        in_feat = in_feat.view(in_feat.size(0), in_feat.size(1), -1).transpose(1,2)
        #[?, M, H*W] @  [？, H*W , Dc]= [?,M, Dc]
        vote_M = torch.bmm(votes, in_feat)
        visual_features = self.relu(vote_M)

      
        return visual_features


#Graph Reasoning Module
#Graph convolution
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight.cuda())
        #support = torch.matmul(input, self.weight)
        adj=adj.to(input.cuda())
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GloRe(nn.Module):
    def __init__(self, in_channels):
        super(GloRe, self).__init__()
        self.N = in_channels // 8
        self.S = in_channels // 1
        
        self.theta = nn.Conv2d(in_channels, self.N, 1, 1, 0, bias=False)
        self.bn_theta = BatchNorm2d(self.N)
        self.relu = nn.ReLU()
        
        self.node_conv = nn.Sequential(nn.Conv1d(self.N, self.N, 1, 1, 0, bias=False),nn.BatchNorm1d(self.N))
        self.channel_conv = nn.Sequential(nn.Conv1d(self.S, self.S, 1, 1, 0, bias=False),nn.BatchNorm1d(self.S))
        
        self.conv_2 = nn.Conv2d(self.S, in_channels, 1, 1, 0, bias=False)
        self.bn3 = BatchNorm2d(in_channels)
        
    def forward(self, x):
        batch, C, H, W = x.size()
        L = H * W
        
        B = self.theta(x).view(-1, self.N, L)
        #B = self.bn_theta(B).view(-1, self.N, L)
        phi = x
        phi = phi.view(-1, self.S, L)
        phi = torch.transpose(phi, 1, 2)

        V = torch.bmm(B, phi) / L
        #V = self.relu(V)
        V = self.relu(self.node_conv(V))
        #V = v1 - V
        #V = self.node_conv(V)
        V = self.relu(self.channel_conv(torch.transpose(V, 1, 2)))
        #V = self.channel_conv(torch.transpose(V, 1, 2))
        
        y = torch.bmm(torch.transpose(B, 1, 2), torch.transpose(V, 1, 2))
        #y = F.softmax(y, dim=1)
        y = y.view(-1, self.S, H, W)
        y = self.conv_2(y)
        y = self.bn3(y)
        return y


#Semantic Mapping Module
class SemanticToLocal(nn.Module):

    def __init__(self, input_feature_channels,  visual_feature_channels):
        super(SemanticToLocal, self).__init__()

        # It is necessary to calculate the mapping weight matrix from 
        #symbol nodes to local features for each image. [?, H*W, M]
        # The W in the paper is as follows
        self.conv1 = nn.Conv2d(input_feature_channels + input_feature_channels, 1,
                              kernel_size=1, stride=1)
        
        self.relu = nn.ReLU(inplace=False)

    def compute_compat_batch(self, batch_input, batch_evolve):
        # batch_input [H, W, Dl]
        # batch_evolve [M, Dc]
        # [H, W, Dl] => [H * W, Dl] => [H*W, M, Dl]
        H = batch_input.shape[0]
        W = batch_input.shape[1]
        M = batch_evolve.shape[0]
        Dl = batch_input.shape[-1]
        batch_input = batch_input.reshape( H * W, Dl)
        batch_input = batch_input.unsqueeze(1).repeat([1,M,1])
        # [M,Dc] => [H*W, M, Dc]
        batch_evolve = batch_evolve.unsqueeze(0).repeat([H*W, 1, 1])
        # [H*W, M, Dc+Dl] 
        batch_concat = torch.cat([batch_input, batch_evolve], axis=-1)
        # [H*W, M, Dc+Dl] =>[1,H*W, M, Dc+Dl]
        batch_concat = batch_concat[np.newaxis,:,:,:]
        # [H*W, M, Dc+Dl] =>[1,Dc+Dl,H*W, M]
        batch_concat = batch_concat.transpose(2,3).transpose(1,2)
        #[1,Dc+Dl,H*W, M] =>[1,1,H*W, M]
        mapping = self.conv1(batch_concat)
        #[1,1,H*W, M] => [1, H*W, M, 1]
        mapping = mapping.transpose(1,2).transpose(2,3)
        #[1,1,H*W, M] => [H*W, M, 1]
        mapping = mapping.view(-1,mapping.size(2),mapping.size(3))
        #[H*W, M,1] => [H*W, M]
        mapping = mapping.view(mapping.size(0), -1)
        mapping = F.softmax(mapping, dim=0)
        return  mapping

    def forward(self, x, evolved_feat):
        # [?, Dl, H, W] , [?, M, Dc]
        input_feat = x 
        evolved_feat = evolved_feat
        # [?, H, W, Dl]
        input_feat = input_feat.transpose(1,2).transpose(2, 3)
        batch_list = []
        for index in range(input_feat.size(0)):
            batch = self.compute_compat_batch(input_feat[index], evolved_feat[index])
            batch_list.append(batch)
        # [?, H*W, M]
        mapping = torch.stack(batch_list, dim=0)
        # [?, M, Dc] => [? * M, Dc] => [? * M, Dl] => [?, M, Dl]
        Dl = input_feat.size(-1)
        M = evolved_feat.size(1)
        H = input_feat.size(1)
        W = input_feat.size(2)
        # [?, M, Dc] => [? * M, Dc]
        #[?, H*W, M] @ [? , M, Dl] => [?, H*W, Dl]
        applied_mapping = torch.bmm(mapping, evolved_feat)
        applied_mapping = self.relu(applied_mapping)
        #[?, H*W, Dl] => [?, H, W, Dl]
        applied_mapping = applied_mapping.reshape(input_feat.size(0), H , W, Dl)
        #[?, H, W, Dl] => [?, Dl, H, W]
        applied_mapping = applied_mapping.transpose(2,3).transpose(1,2)

        return applied_mapping

#overall model layer
class GRMLayer(nn.Module):

    def __init__(self, input_feature_channels,  visual_feature_channels, num_symbol_node,
                 fasttest_embeddings, fasttest_dim, graph_adj_mat):
        super(GRMLayer, self).__init__()

        self.local__to_semantic = LocalToSemantic(input_feature_channels,
                                                             visual_feature_channels,
                                                             num_symbol_node)


        self.graph_reasoning1 = GraphConvolution(300,128)
        self.graph_reasoning2 = GraphConvolution(128,input_feature_channels)

        self.glore = GloRe(input_feature_channels)
        self.final = nn.Sequential(nn.Conv2d(input_feature_channels* 2, input_feature_channels, kernel_size=1, bias=False))
                                   #BatchNorm2d(input_feature_channels))

        self.semantic_to_local = SemanticToLocal(input_feature_channels, visual_feature_channels)
        
        self.graph_adj_mat = torch.FloatTensor(graph_adj_mat).cuda()
        self.visual_feature_channels = visual_feature_channels
        self.fasttest_embeddings = torch.FloatTensor(fasttest_embeddings).cuda()
        self.relu = nn.ReLU(inplace=False)


    def forward(self, x):
        visual_feat = x
        voit = self.local__to_semantic(x)
        
        #fasttest_embeddings = self.fasttest_embeddings.unsqueeze(0)
        #fasttest_embeddings = fasttest_embeddings.repeat(visual_feat.size(0), 1, 1)
        #fasttest_embeddings = fasttest_embeddings.to(visual_feat.cuda())

        fasttest_embeddings = voit
    
        graph_norm_adj = normalize_adjacency(self.graph_adj_mat)
        
        
        batch_list = []
        for index in range(visual_feat.size(0)):
            batch = self.graph_reasoning1(fasttest_embeddings[index], graph_norm_adj)
            batch = F.relu(batch)
            batch_list.append(batch)
        # [?, M, H*W]
        evolved_feat = torch.stack(batch_list, dim=0)
        batch_list1 = []
        for index in range(evolved_feat.size(0)):
            evolved_feats = F.dropout(evolved_feat[index], 0.3)
            batch1 = self.graph_reasoning2(evolved_feats, graph_norm_adj)
            batch1 = F.relu(batch1)
            batch_list1.append(batch1)
        # [?, M, H*W]
        evolved_feat1 = torch.stack(batch_list1, dim=0)
        
        enhanced_feat = self.semantic_to_local(x, evolved_feat1)
        
        out1 = enhanced_feat
        out2 =self.glore(x)

        #out =x + out1 + out2
        #out =out2 + out1
        out = self.final(torch.cat((out1, out2), 1))
        #out = out + x

        return out


class DualGCNHead(nn.Module):

    def __init__(self, input_feature_channels,  visual_feature_channels, num_symbol_node,
                 fasttest_embeddings, fasttest_dim, graph_adj_mat):
        super(DualGCNHead, self).__init__()

        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.final = nn.Sequential(nn.Conv2d(input_feature_channels*2, input_feature_channels, kernel_size=1, bias=False),
                                   BatchNorm2d(input_feature_channels))


        self.dgc = GRMLayer(input_feature_channels,  visual_feature_channels, num_symbol_node,
                 fasttest_embeddings, fasttest_dim, graph_adj_mat)
       


    def forward(self, x):
        gc1 = self.dgc(x)
        x1 = self.maxpool(x)
        gc2 = self.dgc(x1)
        x2 = self.maxpool(x1)
        gc3= self.dgc(x2)
    
        gc3 = gc2 + F.interpolate(gc3, size=(x1.shape[2], x1.shape[3]), mode="bilinear")
        out =gc1 + F.interpolate(gc3, size=(x.shape[2], x.shape[3]), mode="bilinear")
        #out = self.final(torch.cat((out1, out2), 1))
        out =x + out
        #out =x + gc1 + out2
        
        return out


if __name__ == "__main__":
   pass
