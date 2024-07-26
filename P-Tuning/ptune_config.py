# coding:utf-8
import torch


class ProjectConfig(object):
    def __init__(self):
        # self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = 'mps:0'
        self.pre_model = '/Users/ligang/PycharmProjects/llm/prompt_tasks/bert-base-chinese'
        self.train_path = '/Users/ligang/PycharmProjects/llm/prompt_tasks/P-Tuning/data/train.txt'
        self.dev_path = '/Users/ligang/PycharmProjects/llm/prompt_tasks/P-Tuning/data/dev.txt'
        self.verbalizer = '/Users/ligang/PycharmProjects/llm/prompt_tasks/P-Tuning/data/verbalizer.txt'
        self.max_seq_len = 512
        self.batch_size = 8
        self.learning_rate = 5e-5
        # 权重衰减系数
        self.weight_decay = 0
        # 学习率预热的系数
        self.warmup_ratio = 0.06
        self.p_embedding_num = 6
        self.max_label_len = 2
        self.epochs = 50
        self.logging_steps = 10
        self.valid_steps = 20
        self.save_dir = '/Users/ligang/PycharmProjects/llm/prompt_tasks/P-Tuning/checkpoints'


if __name__ == '__main__':
    pc = ProjectConfig()
    print(pc.verbalizer)
