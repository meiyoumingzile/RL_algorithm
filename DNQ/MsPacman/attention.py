import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=1,
                stride=1,
            ),
            nn.ReLU(),  # activation
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=1, stride=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),  # activation
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=1,
                stride=2,
            ),
            nn.ReLU(),  # activation
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=1, stride=1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=1,
                stride=2,
            ),
            nn.ReLU(),  # activation
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=1, stride=1)
        )
        self.lin1 = nn.Linear(32, 4)
        self.relu=nn.ReLU()

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.relu(x)
        return x


class FEAT(nn.Module):
    def __init__(self, args):
        super().__init__()
        hdim=32
        self.cnn=CNN()
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
        self.args=args
    def forward(self, Support,Query,y_spt,y_qrt,mode):

        emb_dim = 32
        # organize support/query data
        support = self.cnn(Support)#10*3*8*8->10*32
        query = self.cnn(Query).resize(1,self.args.k_qry,self.args.n_way,emb_dim)#1,15,2,32
        support =support.resize(1,self.args.k_spt,self.args.n_way,emb_dim)#1,5,2,32

        # get mean of the support
        proto = support.mean(dim=1)  # Ntask x NK x d  2*5*32->2*32
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = self.args.k_qry*self.args.n_way

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        proto = self.slf_attn(proto, proto, proto)
        if self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1)  # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            proto = proto.view(num_batch * num_query, num_proto, emb_dim)  # (Nbatch x Nq, Nk, d)

            logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature

        else:
            proto = F.normalize(proto, dim=-1)  # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim)  # (Nbatch,  Nq*Nw, d)

            logits = torch.bmm(query, proto.permute([0, 2, 1])) / self.args.temperature
            logits = logits.view(-1, num_proto)

        # for regularization
        if mode=='train':
            aux_task = torch.cat([support.view(1, self.args.k_spt, self.args.n_way, emb_dim),
                                  query.view(1, self.args.k_qry, self.args.n_way, emb_dim)], 1)  # T x (K+Kq) x N x d
            num_query = np.prod(aux_task.shape[1:3])
            aux_task = aux_task.permute([0, 2, 1, 3])
            aux_task = aux_task.contiguous().view(-1, self.args.k_spt + self.args.k_qry, emb_dim)
            # apply the transformation over the Aug Task
            aux_emb = self.slf_attn(aux_task, aux_task, aux_task)  # T x N x (K+Kq) x d
            # compute class mean
            aux_emb = aux_emb.view(num_batch, self.args.n_way, self.args.k_spt + self.args.k_qry, emb_dim)
            aux_center = torch.mean(aux_emb, 2)  # T x N x d

            if self.args.use_euclidean:
                aux_task = aux_task.permute([1, 0, 2]).contiguous().view(-1, emb_dim).unsqueeze(
                    1)  # (Nbatch*Nq*Nw, 1, d)
                aux_center = aux_center.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
                aux_center = aux_center.view(num_batch * num_query, num_proto, emb_dim)  # (Nbatch x Nq, Nk, d)

                logits_reg = - torch.sum((aux_center - aux_task) ** 2, 2) / self.args.temperature2
            else:
                aux_center = F.normalize(aux_center, dim=-1)  # normalize for cosine distance
                aux_task = aux_task.permute([1, 0, 2]).contiguous().view(num_batch, -1, emb_dim)  # (Nbatch,  Nq*Nw, d)

                logits_reg = torch.bmm(aux_task, aux_center.permute([0, 2, 1])) / self.args.temperature2
                logits_reg = logits_reg.view(-1, num_proto)

            return logits, logits_reg
        else:
            return logits



class Resnet18_m_classify (nn.Module):
    def __init__(self, model):
        super(Resnet18_m_classify, self).__init__()
        # 取掉model的后两层
        #self.resnet_layer = nn.Sequential(*list(model.children())[:-2])
        self.resnet18 = model
        self.Linear_layer = nn.Linear(1000, 4)

    def forward(self, x):
        x = self.resnet18(x)
        x = self.Linear_layer(x)
        return x



class Resnet18_m_classify_att(nn.Module):
    def __init__(self, model):
        super(Resnet18_m_classify_att, self).__init__()
        # 取掉model的后两层
        #self.resnet_layer = nn.Sequential(*list(model.children())[:-2])
        hdim = 32
        self.resnet18 = model
        self.fc1 =  nn.Linear(1000,32)
        self.attention = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
        self.fc2 =  nn.Linear(32,4)

    def forward(self, x):
        x = self.resnet18(x)
        x = self.fc1(x)
        x = x.unsqueeze(1)
        x = self.attention(x,x,x)
        x = x.squeeze(1)
        x = self.fc2(x)
        return x




# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser(
#         description='arg setting Script')
#     parser.add_argument('--batch_size', default=8, type=int,
#                         help='Batch size for training')
#     parser.add_argument('--resume', default=None, type=str,
#                         help='Checkpoint state_dict file to resume training from. If this is "interrupt"'\
#                              ', the model will resume training from the interrupt file.')
#     parser.add_argument('--start_iter', default=-1, type=int,
#                         help='Resume training at this iter. If this is -1, the iteration will be'\
#                              'determined from the file name.')
#     parser.add_argument('--num_workers', default=4, type=int,
#                         help='Number of workers used in dataloading')
#
#     net = FEAT()
#     print(net)
#     name_list = ['up1', 'up2', 'up3', 'up4', 'up5']  # list中为需要冻结的网络层
#     model = eval(args.model_name)(n_class=n_class)  # 加载model
#     for name, value in model.named_parameters():
#         if name in name_list:
#             value.requires_grad = False
#     params = filter(lambda p: p.requires_grad, model.parameters())
#     optimizer = torch.optim.SGD(params, lr=lr_start, momentum=0.9, weight_decay=0.0005)
