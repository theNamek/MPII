import kenlm
import math
import sys


class PPLEvaluator(object):
    def __init__(self, ppl_model_path):
        self.ppl_model = kenlm.Model(ppl_model_path)

    def cal_ppl(self, texts_transferred):
        """
        :param texts_transferred: list of sentences
        :return:
        """
        sum = 0
        words = []
        length = 0
        for i, line in enumerate(texts_transferred):
            words += [word for word in line.split()]
            length += len(line.split())
            score = self.ppl_model.score(line)
            sum += score
        if 0 == length:
            return 0.0
        return math.pow(10, -sum / length)


if __name__ == '__main__':
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    datas = []
    with open(data_path, encoding='utf-8') as sf:
        for sent in sf:
            sent = sent.strip()
            datas.append(sent)
    evaluator = PPLEvaluator(model_path)
    print(evaluator.cal_ppl(datas))




