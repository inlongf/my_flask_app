import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from transformers import (
    BertModel,
    GPT2LMHeadModel,
    AutoModel,
    AutoTokenizer,
)
from torch_geometric.nn import GCNConv
import snntorch as snn
from snntorch import surrogate
import torch.nn.utils.prune as prune
from config import device  # Import device from config
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class PruningMixin:
    def apply_pruning(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=0.4)
                prune.remove(module, "weight")

### 文本處理應用 ###
class BERT_LSTM_Model(nn.Module, PruningMixin):
    def __init__(self):
        super(BERT_LSTM_Model, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(128, 1)
        self.apply_pruning()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        logits = self.fc(lstm_out[:, -1, :])
        return logits

class FineTunedGPTModel(nn.Module, PruningMixin):
    def __init__(self):
        super(FineTunedGPTModel, self).__init__()
        self.gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.apply_pruning()

    def forward(self, input_ids, attention_mask=None, labels=None):
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
        outputs = self.gpt_model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.logits

class AffectiveModel(nn.Module, PruningMixin):
    def __init__(self, bert_model_name="bert-base-uncased"):
        super(AffectiveModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.emotion_layer = nn.Linear(
            self.bert.config.hidden_size, 8
        )  # 假设有8种情感分类
        self.classification_layer = nn.Linear(
            self.bert.config.hidden_size, 2
        )  # 假设有2类分类任务
        self.apply_pruning()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        emotion_logits = self.emotion_layer(cls_output)
        classification_logits = self.classification_layer(cls_output)
        return emotion_logits, classification_logits

### 圖像處理應用 ###
class GANGenerator(nn.Module):
    def __init__(self):
        super(GANGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.fc(x)

class GANDiscriminator(nn.Module):
    def __init__(self):
        super(GANDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(x)

class AdvancedMultimodalTransformer(nn.Module, PruningMixin):
    def __init__(
        self,
        text_model_name="bert-base-uncased",
        img_model_name="openai/clip-vit-base-patch32",
    ):
        super(AdvancedMultimodalTransformer, self).__init__()
        self.text_model = BertModel.from_pretrained(text_model_name)
        self.img_model = AutoModel.from_pretrained(img_model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.img_tokenizer = AutoTokenizer.from_pretrained(img_model_name)
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=self.text_model.config.hidden_size, num_heads=8
        )
        self.fc = nn.Linear(
            self.text_model.config.hidden_size + self.img_model.config.hidden_size, 1
        )
        self.apply_pruning()

    def forward(self, text_input, image_input):
        text_tokens = self.text_tokenizer(
            text_input, return_tensors="pt", padding=True, truncation=True
        )
        text_output = self.text_model(**text_tokens).last_hidden_state
        img_tokens = self.img_tokenizer(
            image_input, return_tensors="pt", padding=True, truncation=True
        )
        img_output = self.img_model(**img_tokens).last_hidden_state
        combined_output, _ = self.cross_modal_attention(
            text_output, img_output, img_output
        )
        combined_output = torch.cat(
            (text_output[:, -1, :], img_output[:, -1, :]), dim=1
        )
        logits = self.fc(combined_output)
        return logits

### 圖結構數據處理應用 ###
class GCN_GRU_Model(nn.Module, PruningMixin):
    def __init__(self, in_channels, out_channels):
        super(GCN_GRU_Model, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, 16)
        self.gru = nn.GRU(input_size=16, hidden_size=16, num_layers=1, batch_first=True)
        self.fc = nn.Linear(16, out_channels)
        self.apply_pruning()

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x, _ = self.gru(x.view(batch.size(0), -1, 16))
        x = self.fc(x[:, -1, :])
        return x

### 序列生成和變分自編碼器應用 ###
class GRU_VAE(nn.Module, PruningMixin):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GRU_VAE, self).__init__()
        self.encoder_gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc_output = nn.Linear(hidden_dim, input_dim)
        self.apply_pruning()

    def encode(self, x):
        h, _ = self.encoder_gru(x)
        mu = self.fc_mu(h[:, -1, :])
        logvar = self.fc_logvar(h[:, -1, :])
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = z.unsqueeze(1).repeat(1, 10, 1)
        h, _ = self.decoder_gru(z)
        return torch.sigmoid(self.fc_output(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

### 自監督學習應用 ###
class SelfSupervisedRecursiveNN(nn.Module, PruningMixin):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SelfSupervisedRecursiveNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.pred_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.pred_fc2 = nn.Linear(output_dim, hidden_dim)
        self.apply_pruning()

    def forward(self, x):
        hidden = torch.relu(self.fc1(x))
        output = torch.sigmoid(self.fc2(hidden))
        pred_hidden = self.pred_fc1(hidden)
        error_hidden = hidden - pred_hidden
        self.pred_fc1.weight.data += 0.1 * error_hidden.mean(dim=0).unsqueeze(0)
        pred_output = self.pred_fc2(output)
        error_output = hidden - pred_output
        self.pred_fc2.weight.data += 0.1 * error_output.mean(dim=0).unsqueeze(0)
        return output

### 強化學習應用 ###
class LSTM_PolicyNetwork(nn.Module, PruningMixin):
    def __init__(self, state_dim, action_dim):
        super(LSTM_PolicyNetwork, self).__init__()
        self.lstm = nn.LSTM(
            input_size=state_dim, hidden_size=128, num_layers=1, batch_first=True
        )
        self.fc = nn.Linear(128, action_dim)
        self.apply_pruning()

    def forward(self, x):
        h, _ = self.lstm(x)
        action_probs = torch.softmax(self.fc(h[:, -1, :]), dim=-1)
        return action_probs

class SelfSupervisedReinforcementLearningModel(nn.Module, PruningMixin):
    def __init__(self, state_dim, action_dim):
        super(SelfSupervisedReinforcementLearningModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=state_dim, hidden_size=128, num_layers=1, batch_first=True
        )
        self.fc = nn.Linear(128, action_dim)
        self.self_supervised_head = nn.Linear(128, state_dim)
        self.apply_pruning()

    def forward(self, x):
        h, _ = self.lstm(x)
        action_probs = torch.softmax(self.fc(h[:, -1, :]), dim=-1)
        predicted_state = self.self_supervised_head(h[:, -1, :])
        return action_probs, predicted_state

class HierarchicalReinforcementLearningModel(nn.Module, PruningMixin):
    def __init__(self, state_dim, action_dim, subtask_dim):
        super(HierarchicalReinforcementLearningModel, self).__init__()
        self.high_level_policy = nn.LSTM(
            input_size=state_dim, hidden_size=128, num_layers=1, batch_first=True
        )
        self.subtask_policy = nn.ModuleList(
            [nn.Linear(128, subtask_dim) for _ in range(action_dim)]
        )
        self.apply_pruning()

    def forward(self, x):
        h, _ = self.high_level_policy(x)
        subtask_probs = [
            torch.softmax(policy(h[:, -1, :]), dim=-1) for policy in self.subtask_policy
        ]
        return subtask_probs

### 元學習應用 ###
class MetaLearningModel(nn.Module, PruningMixin):
    def __init__(self, base_model):
        super(MetaLearningModel, self).__init__()
        self.base_model = base_model
        self.apply_pruning()

    def forward(self, x):
        return self.base_model(x)

    def meta_update(self, grads):
        for param, grad in zip(self.base_model.parameters(), grads):
            param.data -= self.learning_rate * grad

class MetaPolicyNetwork(nn.Module, PruningMixin):
    def __init__(self, input_dim, action_dim):
        super(MetaPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
        self.apply_pruning()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

class MetaReinforcementLearning:
    def __init__(self, input_dim, action_dim):
        self.policy = MetaPolicyNetwork(input_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)

    def update_policy(self, rewards, log_probs):
        discounted_rewards = [sum(rewards[i:]) for i in range(len(rewards))]
        loss = sum(
            [
                -log_prob * reward
                for log_prob, reward in zip(log_probs, discounted_rewards)
            ]
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def meta_update(self, meta_grads):
        for param, meta_grad in zip(self.policy.parameters(), meta_grads):
            param.data -= 0.01 * meta_grad  # 学习率可以调整

### 記憶增強網絡應用 ###
class MemoryAugmentedNetwork(nn.Module, PruningMixin):
    def __init__(self, input_dim, memory_units, memory_unit_size):
        super(MemoryAugmentedNetwork, self).__init__()
        self.controller = nn.LSTM(input_dim, memory_units, batch_first=True)
        self.memory = nn.Parameter(torch.randn(memory_units, memory_unit_size))
        self.read_head = nn.Linear(memory_units, memory_unit_size)
        self.write_head = nn.Linear(memory_units, memory_unit_size)
        self.attention = nn.MultiheadAttention(embed_dim=memory_unit_size, num_heads=8)
        self.fc = nn.Linear(memory_unit_size, input_dim)
        self.apply_pruning()

    def forward(self, x):
        output, _ = self.controller(x)
        read_weights = torch.softmax(self.read_head(output), dim=-1)
        write_weights = torch.softmax(self.write_head(output), dim=-1)
        read_data = torch.matmul(read_weights, self.memory)
        self.memory = self.memory * (
            1 - write_weights.unsqueeze(-1)
        ) + output.unsqueeze(1) * write_weights.unsqueeze(-1)
        attended_data, _ = self.attention(read_data, read_data, read_data)
        final_output = self.fc(attended_data)
        return final_output

class EnhancedMemoryNetwork(nn.Module, PruningMixin):
    def __init__(self, input_dim, memory_units, memory_unit_size):
        super(EnhancedMemoryNetwork, self).__init__()
        self.controller = nn.LSTM(input_dim, memory_units, batch_first=True)
        self.memory = nn.Parameter(torch.randn(memory_units, memory_unit_size))
        self.read_head = nn.Linear(memory_units, memory_unit_size)
        self.write_head = nn.Linear(memory_units, memory_unit_size)
        self.fc = nn.Linear(memory_unit_size, input_dim)
        self.attention = nn.MultiheadAttention(embed_dim=memory_unit_size, num_heads=8)
        self.apply_pruning()

    def forward(self, x):
        output, _ = self.controller(x)
        read_weights = torch.softmax(self.read_head(output), dim=-1)
        write_weights = torch.softmax(self.write_head(output), dim=-1)
        read_data = torch.matmul(read_weights, self.memory)
        self.memory = self.memory * (
            1 - write_weights.unsqueeze(-1)
        ) + output.unsqueeze(1) * write_weights.unsqueeze(-1)
        attended_data, _ = self.attention(read_data, read_data, read_data)
        final_output = self.fc(attended_data)
        return final_output

### 社會互動分析應用 ###
class SocialInteractionModel(nn.Module, PruningMixin):
    def __init__(self, model_name="bert-base-uncased"):
        super(SocialInteractionModel, self).__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.interaction_layer = nn.Linear(768, 128)
        self.classification_layer = nn.Linear(128, 2)
        self.apply_pruning()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        interaction_output = torch.relu(
            self.interaction_layer(outputs.last_hidden_state[:, 0, :])
        )
        classification_logits = self.classification_layer(interaction_output)
        return classification_logits

### 神經圖靈機應用 ###
class NeuralTuringMachine(nn.Module, PruningMixin):
    def __init__(self, input_dim, output_dim, memory_size, memory_vector_dim):
        super(NeuralTuringMachine, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim

        self.controller = nn.LSTM(input_dim, memory_vector_dim, batch_first=True)
        self.memory = nn.Parameter(torch.randn(memory_size, memory_vector_dim))
        self.read_head = nn.Linear(memory_vector_dim, memory_size)
        self.write_head = nn.Linear(memory_vector_dim, memory_size)
        self.fc = nn.Linear(memory_vector_dim, output_dim)
        self.apply_pruning()

    def forward(self, x):
        controller_out, _ = self.controller(x)
        read_weights = torch.softmax(self.read_head(controller_out[:, -1, :]), dim=-1)
        write_weights = torch.softmax(self.write_head(controller_out[:, -1, :]), dim=-1)
        read_data = torch.matmul(read_weights, self.memory)
        self.memory = self.memory * (1 - write_weights.unsqueeze(-1)) + controller_out[
            :, -1, :
        ].unsqueeze(1) * write_weights.unsqueeze(-1)
        output = self.fc(read_data)
        return output

### 集成學習應用 ###
class Column(nn.Module, PruningMixin):
    def __init__(self, input_dim, output_dim):
        super(Column, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, output_dim)
        self.apply_pruning()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.output_layer(x)

class ThousandBrainsModel(nn.Module, PruningMixin):
    def __init__(self, num_columns, input_dim, output_dim):
        super(ThousandBrainsModel, self).__init__()
        self.columns = nn.ModuleList(
            [Column(input_dim, 128) for _ in range(num_columns)]
        )
        self.global_layer = nn.Linear(128 * num_columns, output_dim)
        self.apply_pruning()

    def forward(self, x):
        column_outputs = [column(x) for column in self.columns]
        combined_output = torch.cat(column_outputs, dim=-1)
        return self.global_layer(combined_output)

class MixtureOfExpertsModel(nn.Module, PruningMixin):
    def __init__(self, expert_list):
        super(MixtureOfExpertsModel, self).__init__()
        self.experts = nn.ModuleList(expert_list)
        self.gating_network = nn.Linear(self.experts[0].input_dim, len(self.experts))
        self.apply_pruning()

    def forward(self, x):
        gate_values = torch.softmax(self.gating_network(x), dim=-1)
        outputs = [
            expert(x) * gate_values[:, i : i + 1]
            for i, expert in enumerate(self.experts)
        ]
        return sum(outputs)

### SNN_LSTM_Model ###
class SNN_LSTM_Model(nn.Module, PruningMixin):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SNN_LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.snn = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.apply_pruning()

    def forward(self, x):
        h, _ = self.lstm(x)
        snn_out, _ = self.snn(h)
        output = self.fc(snn_out[:, -1, :])
        return output

### 数据库模型 ###
class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.Text, nullable=False)
    role = db.Column(db.String(20), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(password)

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class InternetAccessRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    query = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default='pending')
