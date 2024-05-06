import torch
from torch import nn
from torch.nn import functional as F
from transformers import Trainer
from datasets import Dataset
from transformers import RobertaModel, RobertaTokenizer
from transformers import PretrainedConfig, PreTrainedModel
import json
from typing import List, Dict, Callable, Any
import re



class RewardModelConfig(PretrainedConfig):
    model_type = "RewardModel"

    def __init__(self,
                lm_base: str="roberta-base",
                hidden_dims: List[int]=[128, 64, 32],
                activations: List[str]=['relu', 'relu', 'relu'],
                **kwargs: Dict) -> None:
            
        assert 3 == len(hidden_dims), "Current version requires 4 hidden dimensions"
        assert 3 == len(activations), "Current version requires 4 activation functions"
        self.lm_base = lm_base
        self.hidden_dims = hidden_dims
        self.activations = activations
        super().__init__(**kwargs)


class RewardModel(PreTrainedModel):
    config_class = RewardModelConfig

    def __init__(self, config: RewardModelConfig) -> None:
        super().__init__(config)

        ## keep the config
        self.config = config

        ## local device config because depends on runtime
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ## the Large Language Model sent to the device
        self.lm = RobertaModel.from_pretrained(config.lm_base) ## LM is loaded from the config

        ## Tokenizer associated with the LLM
        self.tokenizer = RobertaTokenizer.from_pretrained(config.lm_base)
        
        ## output Linear Model, hidden dims could be fine tuned, 
        ## we don't make it modularizable to much for now, just
        ## output dimensions of the linear layers 
        self.fc1 = nn.Linear(self.lm.config.hidden_size, self.config.hidden_dims[0])
        self.fc2 = nn.Linear(self.config.hidden_dims[0], self.config.hidden_dims[1])
        self.fc3 = nn.Linear(self.config.hidden_dims[1], self.config.hidden_dims[2])
        self.fc4 = nn.Linear(self.config.hidden_dims[2], 1)

        ## used this way because Config can only hold simple inputs such as strings
        self.available_funcs: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
            'relu': nn.functional.relu,
            'tanh': nn.functional.tanh,
            'sigmoid': nn.functional.sigmoid,
            'leaky_relu': nn.functional.leaky_relu,
            'id': lambda x : x,
        }

        for activation in self.config.activations:
            assert activation in self.available_funcs, f"The provided activation : <{activation}> functions are not recognized"
        
        self.activations = self.config.activations + ['id']

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
            Forward pass function. Takes input_ids of one or a batch on input strings
            then pass it through the LM and through the feed forward network on top of it.
        """
        contextual_embedding = self.lm(input_ids).last_hidden_state[:, 0, :]
        
        layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        cur_res = contextual_embedding
        for i, layer in enumerate(layers):
            cur_res = self.available_funcs[self.activations[i]](layer(cur_res))

        return cur_res


    @staticmethod
    def _prepare_given_demo_(tr: str) -> str:
        trs = re.split(r'Assistant:|Human:', tr)[1:]
        human_interactions = []
        assistant_interactions = []
        for i in range(0, len(trs), 2):
            human_interactions.append(trs[i])
            if(i + 1 < len(trs)):
                assistant_interactions.append(trs[i + 1])
        return "[CLS] " + '\n'.join(human_interactions) + " [SEP] " + '\n'.join(assistant_interactions)

    def get_rewards(self, demonstrations: List[Dict[str, str]]) -> List[Dict[str, float]]:
        """
            Function to implement to use with evaluate.py.

            parameters :
                - demonstrations : List of demonstration of the form {'chosen': str, 'rejected': str}
            
            returns :
                - List of entries {'chosen': float, 'rejected': float} with their respective rewards
        """
        outs = []
        for demo in demonstrations:
            
            ## we check if it's our encoding or not :
            if(demo["chosen"].startswith("[CLS]") and
               "[SEP]" in demo["chosen"] and
               demo["rejected"].startswith("[CLS]") and
               "[SEP]" in demo["chosen"]):
                ## It's our encoding we do nothing
                chosen = demo["chosen"]
                rejected = demo["rejected"]
            else :
                ## It's not our encoding, we encode the sentences
                chosen = RewardModel._prepare_given_demo_(demo["chosen"])
                rejected = RewardModel._prepare_given_demo_(demo["rejected"])


            y_plus = torch.tensor(self.tokenizer(chosen, truncation=True, padding='max_length', max_length=512)["input_ids"]).unsqueeze(0)
            y_minus = torch.tensor(self.tokenizer(rejected, truncation=True, padding='max_length', max_length=512)["input_ids"]).unsqueeze(0)

            y_plus_reward = self.forward(y_plus)
            y_minus_reward = self.forward(y_minus)

            outs.append({'chosen': y_plus_reward[0].item(), "rejected": y_minus_reward[0].item()})
        return outs





