import json
import os
with open('../data/MLDS_hw2_data/training_label.json') as json_data:
    d = json.load(json_data)
    path = '../data/parse/'
    for i in range(len(d)):
        qa_path = os.path.join(path, str(i)+'.txt') 
        with open(qa_path, 'w') as test_file:
            for count, item in enumerate(d[i]['caption']):
                test_file.write(item+'\n')
