import torch
import einops

class FourierFeatures(torch.nn.Module):
    def __init__(self, input_dim:int, output_dim:int, learnable:bool = True):
        super(FourierFeatures, self).__init__()

        self.input_dim = input_dim
        self.output_size = output_dim
        self.learnable = learnable
        if self.learnable:
            self.w = torch.nn.Parameter(torch.normal(torch.zeros((self.output_size// 2, input_dim)), torch.ones([self.output_size//2, input_dim]) * 0.2))
        else:
           half_dim = self.output_size // 2
           self.const_w = torch.log(10000) / (half_dim - 1) 
           self.const_w =  torch.exp(torch.arange(half_dim) * -self.const_w)
           self.const_w.requires_grad = False
           self.const_w = torch.nn.Parameter(self.const_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.learnable:
            f = 2 * torch.pi * x * einops.rearrange(self.w, "half_output_size input_dim -> input_dim half_output_size")
        else:
            f = x * self.const_w
        return torch.concatenate([torch.cos(f), torch.sin(f)], axis=-1)

class MLP(torch.nn.Module):
    def __init__(self, input_dims:int, hidden_dims:int, dropout_rate:float = 0.1):
        super(MLP, self).__init__()
        self.hidden_dims = hidden_dims
        self.input_dims =  input_dims
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dims, self.hidden_dims),
            torch.nn.Dropout(dropout_rate),
            torch.nn.LayerNorm(-1),
            torch.nn.SiLU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MLPResNetBlock(torch.nn.Module):
    def __init__(self,feature_dim:int, dropout_rate:float=1.0, use_layer_norm:bool = False, activate_function = torch.nn.ReLU):
        super(MLPResNetBlock, self).__init__()

        self.dropout_rate = dropout_rate
        self.drop_out = torch.nn.Dropout(dropout_rate)
        self.use_layer_norm = use_layer_norm
        self.layer_norm = None

        self.activate_function = activate_function

        self.model = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, feature_dim * 4),
            self.activate_function(),
            torch.nn.Linear(feature_dim * 4, feature_dim)
        )
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        residual = x
        if self.dropout_rate > 0:
            x = self.drop_out(x)

        if self.use_layer_norm:
            x = self.layer_norm(x)
        x = self.model(x)
        return residual + x

class MLPResNet(torch.nn.Module):
    def __init__(self,input_dim:int, num_blocks:int, out_dim:int, hidden_dim:int = 256, dropout_rate:float = 0.1, use_layer_norm:bool = False):

        super(MLPResNet, self).__init__()

        self.input_dim = input_dim
        self.num_blocks = num_blocks
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.first_project = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.activation = torch.nn.SiLU()
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.blocks = torch.nn.ModuleList(
            [MLPResNetBlock(self.hidden_dim, activate_function=torch.nn.SiLU, use_layer_norm=self.use_layer_norm, dropout_rate=self.dropout_rate) for _ in range(self.num_blocks)]
        )
        self.final_project = torch.nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_project(x)
        for block in self.blocks:
            x = block(x)
        x = self.activation(x)
        x = self.final_project(x)
        return x

class ScoreActor(torch.nn.Module):
    def __init__(self,
        visual_input_features:int,
        action_input_features:int,
        output_dim:int,
        num_blocks:int,
        dropout_rate:float,
        hidden_dim:int,
        use_layer_norm:bool,
        time_dim:int = 1,
        ):
        super(ScoreActor, self).__init__()
        self.time_preprocess = FourierFeatures(input_dim=time_dim, output_dim = output_dim)

        self.condition_encoder = None

        self.reverse_network = MLPResNet(
            input_dim=visual_input_features + action_input_features + output_dim,
            num_blocks=num_blocks,
            out_dim=output_dim,
            dropout_rate=dropout_rate,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, obs_enc:torch.Tensor, actions:torch.Tensor, time:torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_enc: (bd..., obs_dim) where bd... is broadcastable to batch_dims
            actions: (batch_dims..., action_dim)
            time: (batch_dims..., 1)
        """
        t_ff = self.time_preprocess.forward(time)

        chunk_size = actions.size(1)
        #cond_enc = einops.repeat(t_ff, "batch action_dim -> batch chunk action_dim", chunk = chunk_size)
        reverse_input = torch.concatenate([t_ff, obs_enc, actions], axis=-1)
        eps_pred = self.reverse_network.forward(reverse_input)
        return eps_pred

class MLPDiffusion(torch.nn.Module):
    def __init__(self, visual_input_dim:int, action_input_dim:int, output_dim:int, time_dim:int, num_blocks:int, dropout_rate:float, hidden_dim:int, use_layer_norm:bool):
        super(MLPDiffusion, self).__init__()

        self.output_dim = output_dim

        self.model = ScoreActor(
            visual_input_features=visual_input_dim,
            action_input_features=action_input_dim,
            output_dim=output_dim,
            num_blocks=num_blocks,
            time_dim=  time_dim,
            dropout_rate=dropout_rate,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
        )


    def compute_loss(self, visual_token:torch.Tensor, action_chunk:torch.Tensor) -> torch.Tensor:
        actions = einops.rearrange(action_chunk, "batch horizon action_dim -> batch (horizon action_dim)")
        assert self.output_dim == actions.size(1)
        batch_size = actions.size(0)
        time = torch.rand((batch_size, 1)).to(actions.device)
        # TODO(noise should be difference with the time)
        noise = torch.randn(actions.size()).to(actions.device)
        noised_actions = actions + noise
        pred_eps = self.model.forward(visual_token, noised_actions, time)
        return torch.nn.MSELoss()(pred_eps, noise)

        