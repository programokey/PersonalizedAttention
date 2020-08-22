import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_unit=32, layers=1,activation_fn=F.leaky_relu):
        super().__init__()
        self.op_list = []
        self.module_list = nn.ModuleList()
        self.linear = nn.Linear(in_features, hidden_unit)
        self.op_list.append(self.linear)
        self.module_list.append(self.linear)
        self.op_list.append(activation_fn)
        for i in range(layers - 1):
            self.linear = nn.Linear(hidden_unit, hidden_unit)
            self.op_list.append(self.linear)
            self.module_list.append(self.linear)
            self.op_list.append(activation_fn)

        self.linear = nn.Linear(hidden_unit, out_features)
        self.op_list.append(self.linear)
        self.module_list.append(self.linear)

    def forward(self, x):
        for layer in self.op_list:
            x = layer(x)
        return x

class PersonalizedAttention(nn.Module):
    def __init__(self,user_num, cat_num, item_num, seller_num, brand_num, embedding_dim=32,
                 hidden_size=32, use_bilinear=True, attention_heads=1,
                 activation_fn=F.leaky_relu):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_embeddings=user_num, embedding_dim=embedding_dim)
        self.cat_embedding = torch.nn.Embedding(num_embeddings=cat_num, embedding_dim=embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=embedding_dim)
        self.seller_embedding = torch.nn.Embedding(num_embeddings=seller_num, embedding_dim=embedding_dim)
        self.brand_embedding = torch.nn.Embedding(num_embeddings=brand_num, embedding_dim=embedding_dim)
        self.activation_fn = activation_fn
        age_ranges_dim = 9
        gender_dim = 3
        self.user_features_transform = MLP(in_features=gender_dim + age_ranges_dim + embedding_dim, out_features=hidden_size)
        self.item_features_transform = MLP(in_features=embedding_dim*4, out_features=hidden_size)
        self.use_bilinear = use_bilinear
        self.attention_head = attention_heads
        if self.use_bilinear:
            self.bilinear = nn.Bilinear(in1_features=hidden_size, in2_features=hidden_size, out_features=attention_heads)



    def forward(self, user_ids, age_ranges, genders, item_ids, cat_ids, seller_ids, brand_ids, mask):
        user_features = torch.cat([self.user_embedding(user_ids), age_ranges, genders], dim=-1)

        item_embedding = self.cat_embedding(item_ids)
        cat_embedding = self.cat_embedding(cat_ids)
        seller_embedding = self.cat_embedding(seller_ids)
        brand_embedding = self.cat_embedding(brand_ids)
        item_features = torch.cat([item_embedding, cat_embedding, seller_embedding,  brand_embedding], dim=-1)
        user_features = self.user_features_transform(user_features)
        item_features = self.item_features_transform(item_features)
        if self.use_bilinear:
            weights = self.bilinear(user_features, item_features)
        else:
            user_features = torch.Tensor()
            item_features = torch.Tensor()
            weights = torch.matmul(user_features.unsqueeze(dim=1).transpose(0, 1), item_features)
        weights = weights*mask
        features = torch.sum(weights*item_features, dim=-2)

