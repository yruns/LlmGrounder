from .semseg import SemanticSegmentationDataset

def build_dataset(**kwargs):
    return SemanticSegmentationDataset(**kwargs)

def build_dataloader(dataset, **kwargs):
    # TODO: add dataloader builder
    pass