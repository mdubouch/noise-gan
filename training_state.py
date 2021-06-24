import torch
import cdc_embedding

class TrainingState():
    """ 
    Holds the objects that need persistency between multiple potential training
    runs, e.g.  the model itself, the optimizer, the total number of epochs,
    the history of loss values, and so on.
    """
    def __init__(self, *model_args, device=torch.device('cpu')):
        self.model = cdc_embedding.CDCEmbedding(*model_args).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.losses = []
        self.test_losses = []
        self.accuracy = []
        self.test_accuracy = []
        self.its = 0
    def save(self, filename):
        torch.save({'state': self}, filename)

