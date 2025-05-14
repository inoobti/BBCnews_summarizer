import datasets

data_src = 'IlyaGusev/gazeta'

class NewsLoader:
    def __init__(self, condition='preproc', src='./data'):
        self.condition = condition
        self.src = src
        if self.condition == 'preproc':
            self.data = datasets.load_from_disk(self.src)
        else:
            self.load_from_github_and_save_to_disc()

    def load_from_github_and_save_to_disc(self):
        data = datasets.load_dataset(self.src)
        data.save_to_disk('./data')

    def load_data(self):
        return self.data


if __name__ == "__main__":
    nl = NewsLoader(condition=None, src=data_src)
    nl.load_from_github_and_save_to_disc()