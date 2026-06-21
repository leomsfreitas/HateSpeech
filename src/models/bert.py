import torch.nn as nn

from transformers import (
    BertModel,
    BertPreTrainedModel
)


class BertForMultiTaskClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = self.classifier(outputs.pooler_output)
        loss = nn.BCEWithLogitsLoss()(logits, labels.float()) if labels is not None else None
        return {"loss": loss, "logits": logits}