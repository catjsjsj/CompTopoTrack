

import math

import torch
import torch.nn as nn
from torch.nn.modules.distance import CosineSimilarity

from src.models.modules.mlp import MLP
import src.models.modules.edge_mpnn as edge_mpnn





class CellTrack_Model(nn.Module):
    

    def __init__(self,
                 hand_NodeEncoder_dic={},
                 learned_NodeEncoder_dic={},
                 intialize_EdgeEncoder_dic={},
                 message_passing={},
                 edge_classifier_dic={}
                 ):
        
        super(CellTrack_Model, self).__init__()

        
        self.distance = CosineSimilarity()

        
        
        self.handcrafted_node_embedding = MLP(**hand_NodeEncoder_dic)
        
        self.learned_node_embedding = MLP(**learned_NodeEncoder_dic)

        
        
        self.learned_edge_embedding = MLP(**intialize_EdgeEncoder_dic)

        
        
        edge_mpnn_class = getattr(edge_mpnn, message_passing.target)
        self.message_passing = edge_mpnn_class(**message_passing.kwargs)

        
        
        self.edge_classifier = MLP(**edge_classifier_dic)

    def forward(self, x, edge_index, edge_feat, motion_edge_feat=None):
        
        
        x1, x2 = x
        x_init = torch.cat((x1, x2), dim=-1)      

        
        src, trg = edge_index
        similarity1 = self.distance(x_init[src], x_init[trg])  
        abs_init = torch.abs(x_init[src] - x_init[trg])        

        
        x1 = self.handcrafted_node_embedding(x1)  
        x2 = self.learned_node_embedding(x2)      
        x = torch.cat((x1, x2), dim=-1)            

        
        src, trg = edge_index
        similarity2 = self.distance(x[src], x[trg])              
        
        
        
        edge_feat_in = torch.cat((
            abs_init,                  
            similarity1[:, None],      
            x[src],                    
            x[trg],                    
            torch.abs(x[src] - x[trg]),
            similarity2[:, None]       
        ), dim=-1)                     
                                        

        
        edge_init_features = self.learned_edge_embedding(edge_feat_in)  

        
        edge_feat_mp = self.message_passing(x, edge_index, edge_init_features)  

        
        pred = self.edge_classifier(edge_feat_mp).squeeze()  

        return pred


class TrackastraObjectFeatureEncoding(nn.Module):
    

    def __init__(self, feature_dim=7, features_per_dim=8, cutoff=1000.0):
        super().__init__()
        if features_per_dim % 2:
            raise ValueError("features_per_dim must be even")
        self.feature_dim = int(feature_dim)
        self.features_per_dim = int(features_per_dim)
        frequencies = torch.exp(
            torch.linspace(0.0, -math.log(float(cutoff)), features_per_dim // 2)
        )
        
        
        self.frequencies = nn.Parameter(
            frequencies.repeat(self.feature_dim, 1)
        )

    @property
    def output_dim(self):
        return self.feature_dim * self.features_per_dim

    def forward(self, features):
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                "Trackastra object features must have shape "
                f"(N, {self.feature_dim}), got {tuple(features.shape)}"
            )
        phase = (
            0.5
            * math.pi
            * features.unsqueeze(-1)
            * self.frequencies.unsqueeze(0)
        )
        encoded = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        encoded = encoded / math.sqrt(self.features_per_dim // 2)
        return encoded.flatten(start_dim=1)


class CellTrack_AssociationModel(nn.Module):
    

    VALID_INPUT_MODES = {
        "embedding_only",
        "embedding_motion",
        "embedding_motion_trackastra",
    }

    def __init__(
        self,
        learned_NodeEncoder_dic,
        edge_encoder_dic,
        message_passing,
        edge_classifier_dic,
        input_mode="embedding_motion",
        motion_feature_dim=4,
        motion_input_indices=(1, 2, 3),
        motion_encoder_dic=None,
        motion_classifier_dic=None,
        motion_residual_scale=1.0,
        object_feature_dim=7,
        object_features_per_dim=8,
        object_feature_cutoff=1000.0,
        object_node_encoder_dic=None,
        object_residual_scale=1.0,
    ):
        super().__init__()
        if input_mode not in self.VALID_INPUT_MODES:
            raise ValueError(
                f"input_mode must be one of {sorted(self.VALID_INPUT_MODES)}, "
                f"got {input_mode!r}"
            )

        self.input_mode = input_mode
        self.motion_feature_dim = int(motion_feature_dim)
        self.motion_input_indices = tuple(int(index) for index in motion_input_indices)
        self.uses_motion = input_mode in {
            "embedding_motion", "embedding_motion_trackastra"
        }
        self.uses_trackastra_objects = input_mode == "embedding_motion_trackastra"
        self.distance = CosineSimilarity()
        self.learned_node_embedding = MLP(**learned_NodeEncoder_dic)

        learned_input_dim = int(learned_NodeEncoder_dic["input_dim"])
        node_dim = int(learned_NodeEncoder_dic["fc_dims"][-1])
        raw_message_kwargs = message_passing.kwargs
        if isinstance(raw_message_kwargs, dict) or hasattr(raw_message_kwargs, "keys"):
            message_kwargs = dict(raw_message_kwargs)
        else:
            message_kwargs = vars(raw_message_kwargs)
        message_node_dim = int(message_kwargs["in_channels"])
        if node_dim != message_node_dim:
            raise ValueError(
                "The learned node encoder output must equal message_passing.in_channels: "
                f"{node_dim} != {message_node_dim}"
            )

        
        
        edge_input_dim = learned_input_dim + 3 * node_dim + 2
        edge_encoder_dic = dict(edge_encoder_dic)
        configured_input_dim = edge_encoder_dic.pop("input_dim", None)
        if configured_input_dim is not None and int(configured_input_dim) != edge_input_dim:
            raise ValueError(
                "edge_encoder_dic.input_dim is inconsistent with the selected input mode: "
                f"{configured_input_dim} != {edge_input_dim}"
            )
        self.edge_input_dim = edge_input_dim
        self.learned_edge_embedding = MLP(input_dim=edge_input_dim, **edge_encoder_dic)

        self.object_feature_encoding = None
        self.object_node_embedding = None
        self.object_node_norm = None
        self.object_residual_scale = None
        if self.uses_trackastra_objects:
            self.object_feature_encoding = TrackastraObjectFeatureEncoding(
                feature_dim=object_feature_dim,
                features_per_dim=object_features_per_dim,
                cutoff=object_feature_cutoff,
            )
            object_node_encoder_dic = dict(
                object_node_encoder_dic
                or {"fc_dims": [64, node_dim], "dropout_p": 0.0, "use_batchnorm": False}
            )
            configured_object_input = object_node_encoder_dic.pop("input_dim", None)
            object_input_dim = self.object_feature_encoding.output_dim
            if configured_object_input is not None and int(configured_object_input) != object_input_dim:
                raise ValueError(
                    "object_node_encoder_dic.input_dim is inconsistent with the "
                    f"Fourier encoding: {configured_object_input} != {object_input_dim}"
                )
            if int(object_node_encoder_dic["fc_dims"][-1]) != node_dim:
                raise ValueError(
                    "The object node encoder output must equal the learned node "
                    f"dimension: {object_node_encoder_dic['fc_dims'][-1]} != {node_dim}"
                )
            self.object_node_embedding = MLP(
                input_dim=object_input_dim, **object_node_encoder_dic
            )
            self.object_node_norm = nn.LayerNorm(node_dim)
            self.object_residual_scale = nn.Parameter(
                torch.tensor(float(object_residual_scale), dtype=torch.float32)
            )

        edge_embedding_dim = int(edge_encoder_dic["fc_dims"][-1])
        if edge_embedding_dim != int(message_kwargs["in_edge_channels"]):
            raise ValueError(
                "The edge encoder output must equal message_passing.in_edge_channels: "
                f"{edge_embedding_dim} != {message_kwargs['in_edge_channels']}"
            )

        self.motion_encoder = None
        self.motion_to_edge = None
        self.motion_classifier = None
        self.motion_residual_scale = None
        if self.uses_motion:
            if not self.motion_input_indices:
                raise ValueError("motion_input_indices must not be empty")
            if min(self.motion_input_indices) < 0 or max(self.motion_input_indices) >= self.motion_feature_dim:
                raise ValueError(
                    "motion_input_indices must index motion_feature_dim="
                    f"{self.motion_feature_dim}, got {self.motion_input_indices}"
                )
            motion_encoder_dic = dict(
                motion_encoder_dic
                or {"fc_dims": [16, 16], "dropout_p": 0.0, "use_batchnorm": False}
            )
            configured_motion_input = motion_encoder_dic.pop("input_dim", None)
            motion_input_dim = len(self.motion_input_indices)
            if configured_motion_input is not None and int(configured_motion_input) != motion_input_dim:
                raise ValueError(
                    "motion_encoder_dic.input_dim is inconsistent with motion_input_indices: "
                    f"{configured_motion_input} != {motion_input_dim}"
                )
            self.motion_encoder = MLP(input_dim=motion_input_dim, **motion_encoder_dic)
            motion_embedding_dim = int(motion_encoder_dic["fc_dims"][-1])
            self.motion_to_edge = nn.Linear(motion_embedding_dim, edge_embedding_dim, bias=False)

            motion_classifier_dic = dict(
                motion_classifier_dic
                or {"fc_dims": [16, 1], "dropout_p": 0.0, "use_batchnorm": False}
            )
            
            
            
            motion_classifier_input_dim = motion_embedding_dim + 2
            if self.uses_trackastra_objects:
                motion_classifier_input_dim += 2
            configured_classifier_input = motion_classifier_dic.pop("input_dim", None)
            if configured_classifier_input is not None and int(configured_classifier_input) != motion_classifier_input_dim:
                raise ValueError(
                    "motion_classifier_dic.input_dim must equal motion embedding plus "
                    "two appearance summaries: "
                    f"{configured_classifier_input} != {motion_classifier_input_dim}"
                )
            self.motion_classifier = MLP(
                input_dim=motion_classifier_input_dim, **motion_classifier_dic
            )
            self.motion_residual_scale = nn.Parameter(
                torch.tensor(float(motion_residual_scale), dtype=torch.float32)
            )

        edge_mpnn_class = getattr(edge_mpnn, message_passing.target)
        self.message_passing = edge_mpnn_class(**message_kwargs)
        self.edge_classifier = MLP(**edge_classifier_dic)

    def forward(self, x, edge_index, edge_feat=None, motion_edge_feat=None):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError("Association model expects (handcrafted_metadata, learned_embedding)")
        handcrafted, learned = x
        if learned.ndim != 2:
            raise ValueError(f"Learned node features must be 2D, got {tuple(learned.shape)}")

        src, trg = edge_index
        raw_similarity = self.distance(learned[src], learned[trg])
        raw_difference = torch.abs(learned[src] - learned[trg])

        nodes = self.learned_node_embedding(learned)
        object_nodes = None
        if self.uses_trackastra_objects:
            encoded_objects = self.object_feature_encoding(
                handcrafted.to(dtype=learned.dtype)
            )
            object_nodes = self.object_node_norm(
                self.object_node_embedding(encoded_objects)
            )
            nodes = nodes + self.object_residual_scale * object_nodes
        node_similarity = self.distance(nodes[src], nodes[trg])
        appearance_parts = [
            raw_difference,
            raw_similarity[:, None],
            nodes[src],
            nodes[trg],
            torch.abs(nodes[src] - nodes[trg]),
            node_similarity[:, None],
        ]

        motion_embedding = None
        if self.uses_motion:
            if motion_edge_feat is None:
                raise ValueError(
                    "embedding_motion requires graph.motion_edge_feat; rebuild the graph cache "
                    "with the current data loader"
                )
            if motion_edge_feat.ndim != 2:
                raise ValueError(
                    f"motion_edge_feat must be 2D, got {tuple(motion_edge_feat.shape)}"
                )
            expected_shape = (edge_index.shape[1], self.motion_feature_dim)
            if tuple(motion_edge_feat.shape) != expected_shape:
                raise ValueError(
                    f"motion_edge_feat must have shape {expected_shape}, "
                    f"got {tuple(motion_edge_feat.shape)}"
                )
            motion_input = motion_edge_feat[:, self.motion_input_indices].to(
                dtype=learned.dtype
            )
            motion_embedding = self.motion_encoder(motion_input)

        edge_input = torch.cat(appearance_parts, dim=-1)
        edge_embedding = self.learned_edge_embedding(edge_input)
        if motion_embedding is not None:
            edge_embedding = edge_embedding + self.motion_to_edge(motion_embedding)
        edge_embedding = self.message_passing(nodes, edge_index, edge_embedding)
        logits = self.edge_classifier(edge_embedding).squeeze(-1)
        if motion_embedding is not None:
            motion_context = torch.cat(
                [
                    motion_embedding,
                    raw_similarity[:, None],
                    raw_difference.mean(dim=-1, keepdim=True),
                ]
                + (
                    [
                        self.distance(object_nodes[src], object_nodes[trg])[:, None],
                        torch.abs(object_nodes[src] - object_nodes[trg]).mean(
                            dim=-1, keepdim=True
                        ),
                    ]
                    if object_nodes is not None
                    else []
                ),
                dim=-1,
            )
            motion_logits = self.motion_classifier(motion_context).squeeze(-1)
            logits = logits + self.motion_residual_scale * motion_logits
        return logits
