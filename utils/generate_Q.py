import sys
import os
import json
import operator
import pickle
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
wordnet = WordNetLemmatizer()
total_file = 1450
film_question_limit = 15
film_line_limit = 30
each_answer_limit = 3
word_count_up = 1000
word_count_down = 10
chars = ['.',',','\'','!']
def cal_count(path):
    wordcount = {}
    id_list = []
    with open(path, 'r') as json_file:
        d = json.load(json_file)
        for item in range(len(d)):
            for cap in d[item]['caption']:
                for c in chars:
                    cap = cap.replace(c,'')
                for word in cap.split():
                    word = word.lower()
                    if word not in wordcount:
                        wordcount[word] = 1
                    else:
                        wordcount[word] += 1
            id_list.append(d[item]['id'])
    '''
    sorted_word = sorted(wordcount.items(), key=operator.itemgetter(1))
    dic_sorted_word = dict(sorted_word)
    print('dic:', dic_sorted_word)
    '''
    return wordcount, id_list
def gen_QA(path, id_list, wordcount):
    question_list = []
    ans_list = []
    _id = []
    for item in range(1450):
        #print(item)
        file_path = os.path.join(path, str(item)+'.txt')
        total_question = {}
        final_question =[]
        final_ans = []
        question_count = 0
        line_count = 0
        with open(file_path,'r') as f:
            for line in f:
                line = line.lower()
                for c in chars:
                    line = line.replace(c,'')
                if question_count >= film_question_limit or line_count >= film_line_limit:
                    break
                question = line.split('\t')[0]
                answer_phrase = line.split('\t')[2]
                ###finding WH- in  {What, How many}
                tagged_sentence = pos_tag(answer_phrase.split())
                if question.split(' ')[0] == 'what': #find 'NN' or 'NNS'
                    for word, pos in tagged_sentence:
                        if pos == 'NNS':
                            index = word
                            ans = wordnet.lemmatize(word)
                        elif pos == 'NN':
                            index = word
                            ans = word
                elif question.split(' ')[0] == 'how': #find 'CD'
                    for word, pos in tagged_sentence:
                        if pos == 'CD':
                            index = word
                            ans = word
                            break
                else:
                    continue
                if ans not in total_question:
                    total_question[ans] = 1
                else:
                    total_question[ans] += 1
                try:
                    count = wordcount[index]
                except KeyError:
                    count = 0
                if total_question[ans] <= each_answer_limit and count < word_count_up and count >= word_count_down:
                    final_question.append(question)
                    final_ans.append(ans)
        ans_list.append(final_ans)
        question_list.append(final_question)
        _id.append(id_list[item])
    print('total qa pair:', sum(len(x) for x in ans_list))
    ans_dict = {}
    for film_ans_set in ans_list:
        for ans in film_ans_set:
            ans_dict[ans] = ans_dict.get(ans, 0) + 1
    #[ans_dict[ans]=ans_dict.get(ans, 0) + 1 for ans in film_ans_set for film_ans_set in ans_list]
    print('ans_dict:',ans_dict)
    how_many_count = 0
    what_count = 0
    for final_question in question_list:
        for question in final_question:
            if question.split(' ')[0] == 'how':
                how_many_count += 1
            elif question.split(' ')[0] == 'what':
                what_count += 1
    print('how many',  how_many_count)
    print('what:', what_count)
    '''
    import matplotlib.pyplot as plt
    a = [i for i in range(1, len(ans_dict)+1)]
    b = list(ans_dict.values())
    print('a:', len(a))
    print('b:', len(b))
    plt.figure()
    plt.scatter(a,b,s = 10)
    plt.savefig('test.png', dpi=300)
    '''
    #print(ans_list)
    '''
    with open('../data/ans_list.pkl','wb') as f:
        pickle.dump(ans_list, f)
    with open('../data/ques_list.pkl','wb') as f:
        pickle.dump(question_list, f)
    with open('../data/_id.pkl','wb') as f:
        pickle.dump(_id, f)
    '''

wordcount, id_list = cal_count('../data/MLDS_hw2_data/training_label.json')
gen_QA('../data/QAset/', id_list, wordcount)
