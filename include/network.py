from include.settings import getSettings
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# include/network.py  (add this class next to your existing ANN Actor)
# include/network.py  — add below your imports and ANN Actor



class ActorSNN(nn.Module):
    """
    SNN version of the simple ANN actor:
      ANN:  Linear(S->H) -> LeakyReLU -> Linear(H->H) -> LeakyReLU -> Linear(H->1) -> Tanh
      SNN:  Linear(S->H) -> LIF  -> Linear(H->H) -> LIF  (unrolled T steps)
            readout uses avg firing-rate of the last LIF -> Linear(H->1) -> Tanh

    Settings (JSON keys; with safe fallbacks):
      - number_of_layer     (int, default 2)
      - snn_unit            (int, default settings['actor_nn'])
      - number_of_timestep  (int, default 8)
      - v_threshold         (float, default 1.0)
      - v_reset             (float, default 0.0)
      - learn_beta          (bool,  default True)
      - learn_threshold     (bool,  default True)
      - reset_mechanism     (str,   default "subtract")
      - sparsity_lambda     (float, default 0.0)  # optional regularizer
    """
    def __init__(self, state_dim, settings=getSettings()):
        super().__init__()

        # --- hyperparams from settings (with ANN-aligned fallbacks) ---
        self.T  = int(settings.get("number_of_timestep", 8))
        self.L  = int(max(1, settings.get("number_of_layer", 2)))
        self.H       = int(settings.get("snn_unit", settings.get("actor_nn", 128)))

        v_th    = float(settings.get("v_threshold", 1.0))
        v_reset = float(settings.get("v_reset", 0.0))
        learn_b = bool(settings.get("learn_beta", True))
        learn_t = bool(settings.get("learn_threshold", True))
        reset_m = settings.get("reset_mechanism", "subtract")
        self.sparsity_lambda = float(settings.get("sparsity_lambda", 0.0))
        self.state_dim = state_dim

        spike_grad = surrogate.atan()  # stable default

        # --- layers: Linear -> LIF (x2 by default) ---
        self.fc_in  = nn.Linear(state_dim, self.H)
        self.lif_in = snn.Leaky(
            beta=torch.ones(self.H) * 0.1,
            threshold=torch.ones(self.H) * v_th,
            reset_mechanism=reset_m,
            learn_beta=learn_b,
            learn_threshold=learn_t,
            spike_grad=spike_grad,
        )

        self.fc_h   = nn.ModuleList()
        self.lif_h  = nn.ModuleList()
        for _ in range(self.L - 1):
            self.fc_h.append(nn.Linear(self.H, self.H))
            self.lif_h.append(
                snn.Leaky(
                    beta=torch.ones(self.H) * 0.1,
                    threshold=torch.ones(self.H) * v_th,
                    reset_mechanism=reset_m,
                    learn_beta=learn_b,
                    learn_threshold=learn_t,
                    spike_grad=spike_grad,
                )
            )

        # readout: avg last-layer rate -> Linear -> Tanh  (=> action in [-1,1])
        self.readout = nn.Linear(self.H, 1)
        self.out_act = nn.Tanh()

        # diagnostics for optional sparsity loss
        self.register_buffer("_last_avg_rate", torch.tensor(0.0))

        # init like your ANN (Kaiming/He)
        for m in [self.fc_in, *self.fc_h, self.readout]:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, state, *_, **__):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        B, device = state.size(0), state.device

        # ---- FIX: initialize membranes as zeros (version-agnostic) ----
        mem_in = torch.zeros(B, self.H, device=device)
        mem_h  = [torch.zeros(B, self.H, device=device) for _ in self.lif_h]

        spike_sum_last = None
        layer_spike_acc = 0.0
        num_layers = 1 + len(self.lif_h)

        for _ in range(self.T):
            h = self.fc_in(state)
            spk, mem_in = self.lif_in(h, mem_in)
            h = spk
            layer_spike_acc += spk.mean()

            for i, (fc, lif) in enumerate(zip(self.fc_h, self.lif_h)):
                h = fc(h)
                spk, mem_h[i] = lif(h, mem_h[i])
                h = spk
                layer_spike_acc += spk.mean()

            spike_sum_last = h if spike_sum_last is None else (spike_sum_last + h)

        rate_last = spike_sum_last / float(self.T)
        with torch.no_grad():
            self._last_avg_rate = layer_spike_acc / (num_layers * self.T)

        out = self.readout(rate_last)
        return self.out_act(out)
    

    def calculate_energy(self):

        E_ACs = 0.9
        E_MACs = 4.6

        fr = self._last_avg_rate

        macs = (self.state_dim*self.H)*self.T

        acs = 0 

        for i, (fc, lif) in enumerate(zip(self.fc_h, self.lif_h)):

            acs += (self.H**2*fr)*self.T

        acs += (self.H * fr)*self.T

        energy = E_ACs * acs + E_MACs * macs

        return energy


    def sparsity_loss(self):
        """Optional spike-rate regularizer (add to actor loss if desired)."""
        if self.sparsity_lambda <= 0:
            return torch.zeros((), device=self._last_avg_rate.device)
        return self.sparsity_lambda * self._last_avg_rate



class Actor(nn.Module):
    def __init__(self, state_dim, settings = getSettings()):
        super(Actor, self).__init__()
        
        self.lrelu_alpha = settings['lrelu_alpha']
        actor_nn = settings['actor_nn']

        self.input = nn.Linear(state_dim, actor_nn)
        self.hidden = nn.Linear(actor_nn, actor_nn)
        self.output = nn.Linear(actor_nn, 1)

    def forward(self, state):
        x = F.leaky_relu(self.input(state), self.lrelu_alpha)
        x = F.leaky_relu(self.hidden(x), self.lrelu_alpha)
        x = torch.tanh(self.output(x))

        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, settings = getSettings()):
        super(Critic, self).__init__()
        
        self.lrelu_alpha = settings['lrelu_alpha']
        critic_nn = settings['critic_nn']
                
        self.input = nn.Linear(state_dim + action_dim, critic_nn)
        self.hidden1 = nn.Linear(critic_nn, critic_nn)
        self.hidden2 = nn.Linear(critic_nn, critic_nn)
        self.output = nn.Linear(critic_nn, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.leaky_relu(self.input(x), self.lrelu_alpha)
        x = F.leaky_relu(self.hidden1(x), self.lrelu_alpha)
        x = F.leaky_relu(self.hidden2(x), self.lrelu_alpha)
        x = self.output(x)
        
        return x
    
class CriticSNN(nn.Module):
    """
    SNN version of the simple ANN critic:
    ANN: Linear(S+A->H) -> LeakyReLU -> Linear(H->H) -> LeakyReLU -> Linear(H->H) -> LeakyReLU -> Linear(H->1)
    SNN: Linear(S+A->H) -> LIF -> Linear(H->H) -> LIF -> Linear(H->H) -> LIF (unrolled T steps)
    readout uses avg firing-rate of the last LIF -> Linear(H->1)
    
    Settings (JSON keys; with safe fallbacks):
    - number_of_layer (int, default 3)  # including input layer
    - snn_unit (int, default settings['critic_nn'])
    - number_of_timestep (int, default 8)
    - v_threshold (float, default 1.0)
    - v_reset (float, default 0.0)
    - learn_beta (bool, default True)
    - learn_threshold (bool, default True)
    - reset_mechanism (str, default "subtract")
    - sparsity_lambda (float, default 0.0)  # optional regularizer
    - lrelu_alpha (float, default settings['lrelu_alpha'])  # for potential fallback, but not used in SNN
    """
    def __init__(self, state_dim, action_dim, settings=getSettings()):
        super().__init__()
        # --- hyperparams from settings (with ANN-aligned fallbacks) ---
        self.T = int(settings.get("number_of_timestep", 8))
        self.L = int(max(1, settings.get("critic_number_of_layer", 3)))
        self.H = int(settings.get("snn_unit", settings.get("critic_nn", 128)))
        v_th = float(settings.get("v_threshold", 1.0))
        v_reset = float(settings.get("v_reset", 0.0))
        learn_b = bool(settings.get("learn_beta", True))
        learn_t = bool(settings.get("learn_threshold", True))
        reset_m = settings.get("reset_mechanism", "subtract")
        self.sparsity_lambda = float(settings.get("sparsity_lambda", 0.0))
        self.lrelu_alpha = settings.get("lrelu_alpha", 0.01)  # kept for compatibility, not used in forward

        spike_grad = surrogate.atan()  # stable default; assumes snn.surrogate is available

        # --- input layer: Linear(S+A -> H) -> LIF ---
        self.fc_in = nn.Linear(state_dim + action_dim, self.H)
        self.lif_in = snn.Leaky(
            beta=torch.ones(self.H) * 0.1,
            threshold=torch.ones(self.H) * v_th,
            reset_mechanism=reset_m,
            learn_beta=learn_b,
            learn_threshold=learn_t,
            spike_grad=spike_grad,
        )

        # --- hidden layers: list of Linear(H -> H) -> LIF (L-1 layers) ---
        self.fc_h = nn.ModuleList()
        self.lif_h = nn.ModuleList()
        for _ in range(self.L - 1):
            self.fc_h.append(nn.Linear(self.H, self.H))
            self.lif_h.append(
                snn.Leaky(
                    beta=torch.ones(self.H) * 0.1,
                    threshold=torch.ones(self.H) * v_th,
                    reset_mechanism=reset_m,
                    learn_beta=learn_b,
                    learn_threshold=learn_t,
                    spike_grad=spike_grad,
                )
            )

        # --- readout: avg last-layer rate -> Linear(H -> 1) ---
        self.readout = nn.Linear(self.H, 1)

        # diagnostics for optional sparsity loss
        self.register_buffer("_last_avg_rate", torch.tensor(0.0))

        # init like your ANN (Kaiming/He)
        for m in [self.fc_in, *self.fc_h, self.readout]:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
        B, device = state.size(0), state.device

        # ---- initialize membranes as zeros ----
        mem_in = torch.zeros(B, self.H, device=device)
        mem_h = [torch.zeros(B, self.H, device=device) for _ in self.lif_h]

        spike_sum_last = None
        layer_spike_acc = 0.0
        num_layers = 1 + len(self.lif_h)

        x = torch.cat([state, action], dim=1)  # input: state + action

        for _ in range(self.T):
            # input layer
            h = self.fc_in(x)
            spk, mem_in = self.lif_in(h, mem_in)
            h = spk  # pass spikes to next
            layer_spike_acc += spk.mean()

            # hidden layers
            for i, (fc, lif) in enumerate(zip(self.fc_h, self.lif_h)):
                h = fc(h)
                spk, mem_h[i] = lif(h, mem_h[i])
                h = spk  # pass spikes
                layer_spike_acc += spk.mean()

            # accumulate last layer spikes for readout
            if spike_sum_last is None:
                spike_sum_last = h
            else:
                spike_sum_last += h

        rate_last = spike_sum_last / float(self.T)

        with torch.no_grad():
            self._last_avg_rate = layer_spike_acc / (num_layers * self.T)

        out = self.readout(rate_last)
        return out

    def sparsity_loss(self):
        """
        Optional spike-rate regularizer (add to critic loss if desired).
        """
        if self.sparsity_lambda <= 0:
            return torch.zeros((), device=self._last_avg_rate.device)
        return self.sparsity_lambda * self._last_avg_rate