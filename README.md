# NLP_with_HF_Transformers

## Let's Practice

### Exercise 1 - Sentiment Analysis
In this Exercise, use "cardiffnlp/twitter-roberta-base-sentiment" model pre-trained on tweets data, to analyze any tweet of choice. Optionally, use the default model (used in Example 1) on the same tweet, to see if the result will change.

```python
from transformers import pipeline

specific_model = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")
data = "Artificial intelligence and automation are already causing friction in the workforce. Should schools revamp existing programs for topics like #AI, or are new research areas required?"
specific_model(data)
```
```python
original_model = pipeline("sentiment-analysis")
data = "Artificial intelligence and automation are already causing friction in the workforce. Should schools revamp existing programs for topics like #AI, or are new research areas required?"
original_model(data)
```
Result :
```python
Device set to use cuda:0
[{'label': 'LABEL_1', 'score': 0.5272253155708313}]
```
```python
No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision 714eb0f (https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
Using a pipeline without specifying a model name and revision in production is not recommended.
Device set to use cuda:0
[{'label': 'NEGATIVE', 'score': 0.9989722967147827}]

```
### Exercise 2 - Topic Classification
In this Exercise, use any sentence of choice to classify it under any classes/ topics of choice. Use "zero-shot-classification" and specify the model="facebook/bart-large-mnli".

```python
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

result = classifier(
    "I love travelling and learning new cultures",
    candidate_labels=["art", "education", "travel"],
)
print(result)
```
Result :
```python
Device set to use cuda:0
{'sequence': 'I love travelling and learning new cultures', 'labels': ['travel', 'education', 'art'], 'scores': [0.9902300238609314, 0.005778131075203419, 0.003991869743913412]}
```
### Exercise 3 - Text Generation Models
In this Exercise, use 'text-generator' and 'gpt2' model to complete any sentence. Define any desirable number of returned sentences.

```python
generator = pipeline('text-generation', model='gpt2')

output = generator("Hello, I'm a language model", max_length=30, num_return_sequences=3)
for i, sentence in enumerate(output):
    print(f"Generated {i+1}: {sentence['generated_text']}")

```
Result :
```python
I'm a model of the language model.

I'm a model of my interaction with the language.

I'm a model of my interaction with the language.

I'm a model of my interaction with the language.

I'm a model of my interaction with the language.

I'm a model of my interaction with the language.

I'm a model of my interaction with the language.

I'm a model of my interaction with the language.

I'm a model of my interaction with the language.

I'm a model of my interaction with the language.

I'm a model of my interaction with the language.

I'm a model of my interaction with the language.

I'm a model of my interaction with the language.

I'm a model of my interaction with the language.

I'm a model of my interaction with the language.

I'm a model of my interaction with the language.

I'm a model of my interaction with the language.

I'm a model of my interaction with the language.

I'm a model of my interaction with the language.

I'm a model of my interaction with
Generated 3: Hello, I'm a language modeler for your application. I want to write a code that can use your code by you. I just want to be able to use your code for my own purposes.

So, I'm going to use something called a simple function that's going to execute a function that takes a dictionary of objects. This is an easy way to write a simple function.

I wanted to use this simple function to run the code that I created. It's pretty simple. Just add a function that takes a dictionary.

def run-function ( dictionary ): return { "dict": dictionary, }

I'm going to have the dictionary in my application and I'm going to run the code that I created. Now, I'm going to have to do some tests.

I'm going to start the tests by taking a dictionary and having the app open up. I'm going to have the app open up by opening up a file and then I'm going to have the app open up by opening up a new file.

I hope you like the code. I really hope you like the code. This is a very simple example. Let's see how it works.

We're going to start by opening up a file.

We
```
### Exercise 4 - Name Entity Recognition
In this Exercise, use any sentence of choice to extract entities: person, location and organization, using Name Entity Recognition task, specify model as "Jean-Baptiste/camembert-ner".

```python
nlp = pipeline("ner", model="Jean-Baptiste/camembert-ner", grouped_entities=True)
example = "Her name is Anjela and she lives in Seoul."

ner_results = nlp(example)
print(ner_results)
```
Result :
```python
Device set to use cuda:0
[{'entity_group': 'PER', 'score': np.float32(0.9481442), 'word': 'Anjela', 'start': 11, 'end': 18}, {'entity_group': 'LOC', 'score': np.float32(0.9986114), 'word': 'Seoul', 'start': 35, 'end': 41}]
```
### Exercise 5 - Question Answering
In this Exercise, use any sentence and a question of choice to extract some information, using "distilbert-base-cased-distilled-squad" model.

```python
question_answerer = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question_answerer(
    question="Which lake is one of the five Great Lakes of North America?",
    context="    Lake Ontario is one of the five Great Lakes of North America. It is surrounded on the north, west, and southwest by the Canadian province of Ontario, and on the south and east by the U.S. state of New York,whose water boundaries, along the international border, meet in the middle of the lake."
    )
```
Result :
```python
Device set to use cuda:0
{'score': 0.9834363460540771, 'start': 4, 'end': 16, 'answer': 'Lake Ontario'}
```
### Exercise 6 - Text Summarization
In this Exercise, use any document/paragraph of choice and summarize it, using "sshleifer/distilbart-cnn-12-6" model.

```python
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6",  max_length=59)
summarizer(
    """
Lake Superior in central North America is the largest freshwater lake in the world by surface area and the third-largest by volume, holding 10% of the world's surface fresh water. The northern and westernmost of the Great Lakes of North America, it straddles the Canada–United States border with the province of Ontario to the north, and the states of Minnesota to the northwest and Wisconsin and Michigan to the south. It drains into Lake Huron via St. Marys River and through the lower Great Lakes to the St. Lawrence River and the Atlantic Ocean.
"""
)
```
Result :
```python
Device set to use cuda:0
Your max_length is set to 142, but your input_length is only 118. Since this is a summarization task, where outputs shorter than the input are typically wanted, you might consider decreasing max_length manually, e.g. summarizer('...', max_length=59)
[{'summary_text': " Lake Superior is the largest freshwater lake in the world by surface area . It holds 10% of the world's surface fresh water . It straddles the Canada–U.S. border with the province of Ontario to the north . It drains into Lake Huron via St. Marys River and through the lower Great Lakes to the St. Lawrence River and the Atlantic Ocean ."}]
```
### Exercise 7 - Translation
In this Exercise, use any sentence of choice to translate English to German. The translation model you can use is "translation_en_to_de".

```python
translator = pipeline("translation_en_to_de", model="t5-small")
print(translator("New York is my favourite city", max_length=40))
```
Result :
```python
Device set to use cuda:0
Both `max_new_tokens` (=256) and `max_length`(=40) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
[{'translation_text': 'New York ist meine Lieblingsstadt'}]
```
## Congratulations! You have completed this guided project.
