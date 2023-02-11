# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division  # use float division as default format
from __future__ import print_function

import sys
import json


if __name__ == '__main__':
    in_path = str(sys.argv[1])
    out_path = str(sys.argv[2])
    with open(in_path, encoding='utf-8') as srcf, open(out_path, 'wt', encoding='utf-8') as tgtf:
        for line in srcf:
            item = json.loads(line)
            exp = item['gen_exp']

            tgtf.write('{}\n'.format(exp))
            tgtf.flush()


