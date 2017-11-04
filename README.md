# VideoQA
Video Question Answering


Dir Structure:

VideoQA/
---data/
   ---MLDS_hw2_data
   ---parse/
      ---0.txt ~ 1449.txt
   ---QAset/
      ---0.txt ~ 1449.txt
---utils/
   ---json_paarse.txt (parse from training_label.json)
   ---generate_Q.py (generate QA list from data/QAset/)
   ---read_json.py (transform caption to QA from data/ parse/)

run:
   python generate_Q.py
