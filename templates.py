SYSTEM_TEMPLATE = \
"""You're an AI agent who is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."""

HUMAN_TEMPLATE = \
"""
You're a consistency scoring agent which is given a set consisting of  two (input, output) pairs in the form [Pair1: (Input1, Output1), Pair2: (Input2, Output2)]. You're also given the type of (input, output) pairs like "Short form Question-Answering", "Translation", "Long form Question-Answering", etc. The inputs in the set are exactly same or might be paraphrases of each other. 
*CONSISTENCY*:  A set is said to be consistent if the two outputs convey the *EXACT* same meaning. There might be more than one common *FACT* in the pair but the pair is said to be consistent *ONLY IF* all the facts have an exact match.
*NOTE*: You should only check if the two outputs convey the same thing.
                   You should ignore anything about correctness of an outputs w.r.t. the input.

set: [
"Pair1": (
"Input1": "{input1}",
"Output1": "{output1}",
),
"Pair2": (
"Input2": "{input2}",
"Output2": "{output2}",
),
"Type": "{Type}", 
"Output Consistency": 
]

*INSTRUCTION*: Based on the above set -
1. Thoroughly check if Output1 and Output2 are consistent with *REASONING* and *EXPLAINING* every part.
2. At the end, you must return the updated set with filling up the "Output Consistency" with "Yes" or "No".
"""

PP_TEMPLATE = \
"""
Today I want you to learn the ways of paraphrasing a sentence or a question. Below are few methods with examples. Go through them carefully.

1. Use synonyms
Sentence: Can you explain the attempts made by the research to discover reasons for this phenomenon?
Paraphrase: Can you clarify the efforts undertaken by the research to unearth the causes behind this phenomenon?

2. Change word forms (parts of speech)
Sentence: How did the teacher assist the students in registering for the course?
Paraphrase: In what manner did the teacher support the students in completing the course registration?

3. Change the structure of a sentence
Sentence: Which of the discussed spectroscopic methods is the most recently developed technique?
Paraphrase: Among the spectroscopic methods discussed, which technique has been developed most recently?

4. Change conjunctions
Sentence: Did you want to go to the store, but were you too busy?
Paraphrase: Although you were busy, did you still want to go to the store?

5. Use idioms
Sentence: How would you describe his emotional state as being very sad?
Paraphrase: How would you depict him feeling down in the dumps?

Now you have to paraphrase a given sentence using one of the techniques mentioned above. I will provide you the number of the technique to use.
Technique Number: {method}
Sentence: {sentence}
Paraphrase:"""


EVAL_STEP1_TEMPLATE = \
"""
Context: The answer to this question depends on the circumstances. Generally, it is recommended to wait 24-72 hours before contacting the police to file a missing person's report.
Question: How much hour to wait before contacting the police.
Answer using the above context. Give answer in the least possible words.
Answer: 24-72 hours.

Context: Nothing happens if you eat watermelon seeds. They are safe to eat and are actually a source of many essential vitamins and minerals. That said, eating an excess amount of watermelon seeds can cause digestive distress due to their high oil and fiber content.
Question: What happens if you eat watermelon seeds?
Answer using the above context. Give answer in the least possible words.
Answer: Nothing; they are safe to eat.

Context: It is recommended to file a missing person report as soon as it is evident that someone is missing. Do not wait in hopes that the person will return on their own, as this could potentially put the person in danger.
Question: How much hour to wait before contacting the police.
Answer using the above context. Give answer in the least possible words.
Answer: As soon as possible.

Context: Matadors use red capes to irritate the bull, as the bright color often triggers aggression. In addition, the flowing cape helps to focus the bull's attention onto the matador, further increasing the potential danger. Red also symbolizes strength, passion, and power.
Question: What do matadors do with red capes?
Answer using the above context. Give answer in the least possible words.
Answer: Irritate and focus the bull's attention.

Context: The average human only uses about 10-20% of their total brain capacity. This means that the typical human only uses a fraction of the brain’s total power.
Question: How much of their total brain capacity do humans usually use?
Answer using the above context. Give answer in the least possible words.
Answer: 10-20%.

Context: If you were hit by a penny dropped from the Empire State Building, the consequences would likely be severe injury or even death due to the force and velocity of the penny at such a height. The penny, which weighs 2.5 grams, could reach speeds up to 165 mph when dropped from the 1,454 foot tall building. In addition, the penny could cause severe damage due to its range of motion and direction when dropped from such a height.
Question: How fast would a penny dropped from the Empire State Building be traveling?
Answer using the above context. Give answer in the least possible words.
Answer: 165 mph.

Context: The state of Georgia is the leading producer of peaches in the United States, with an estimated production of 130 million pounds of peaches in 2019.
Question: Which state produces the most peaches in the US?
Answer using the above context. Give answer in the least possible words.
Answer: Georgia.

Context: {context}
Question: {question}"""


EVAL_STEP2_TEMPLATE = \
"""
Question: How much hour to wait before contacting the police?
Answer 1: As soon as possible.
Answer 2: 24-72 hours.
Are both of the answers same?
No

Question: What do matadors do with red capes?
Answer 1: Wave to anger and distract the bull.
Answer 2: Irritate and focus the bull's attention.
Are both of the answers same? 
No

Question: What happens if you eat watermelon seeds?
Answer 1: Nothing
Answer 2: Nothing; they are safe to eat.
Are both of the answers same?
Yes

Question: How much of their total brain capacity do humans usually use?
Answer 1: 10-20%.
Answer 2: Virtually all of it.
Are both of the answers same?
No

Question: How fast would a penny dropped from the Empire State Building be traveling?
Answer 1: 125 mph.
Answer 2: 165 mph. 
Are both of the answers same?
No

Question: Which state produces the most peaches in the US?
Answer 1: Georgia.
Answer 2: Georgia.
Are both of the answers same?
Yes

Question: {question}
Answer 1: {answer1}
Answer 2: {answer2}
Are both of the answers same?
"""
     