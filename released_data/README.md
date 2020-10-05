- Original Annotated Files, train/dev/test.json:
  ```
  {
      "question_id": "00153f694413a536",
      "question": "What is the middle name of the player with the second most National Football League career rushing yards ?",
      "table_id": "List_of_National_Football_League_rushing_yards_leaders_0",
      "answer-text": "Jerry",
      "question_postag": "WP VBZ DT JJ NN IN DT NN IN DT JJ RBS NNP NNP NNP NN VBG NNS ."
  }
  ```

- Files with traced answer, train/dev.traced.json: note that the traced answer node is approximated, not guaranteed to be correct. We use these as weakly supervised label to train our model.
```
  {
    "question_id": "00153f694413a536",
    "question": "What is the middle name of the player with the second most National Football League career rushing yards ?",
    "table_id": "List_of_National_Football_League_rushing_yards_leaders_0",
    "answer-text": "Jerry",
    "question_postag": "WP VBZ DT JJ NN IN DT NN IN DT JJ RBS NNP NNP NNP NN VBG NNS ."
    "answer-node": [node1, node2, ...]
  }
```
