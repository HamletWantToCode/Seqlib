from .lang_dataset import LangDataset
from ..base import BaseDataLoader


class LangDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, lang1, lang2, batch_size, mode='train', shuffle=True, validation_split=0.0, num_workers=1):
        if mode == 'train':
            root_dir = data_dir + '/data/train'
        else:
            root_dir = data_dir + '/data/test'
        
        dataset = LangDataset(root_dir, lang1, lang2)
        self.input_n_words = dataset.input_lang.n_words
        self.output_n_words = dataset.output_lang.n_words
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)
