# Task 1 failure analysis

All samples come from `checkpoints/best.pt` (epoch 10, val loss 0.557) and are saved in `outputs/samples.txt`. Five prompts were sampled at temperature 0.8 with no top-k, and two were decoded greedily, 600 new characters each.

## Failure case 1: loss of coherence (characters merge into one role)

**Prompt:** "The dog was very sad because" (temperature 0.8)

**Snippet:**

```
The dog was very sad because his hat was ruined. His mom took him to the doctor to show him some money to the doctor. The doctor told the doctor that he would be gone to buy his hat. The doctor said he would get a shot for the doctor to give him a treat.
...
The doctor was very happy and forgave the doctor. He said that he was brave and could not wait to go home again. The doctor told Lily what she had done
```

**Failure type:** loss of coherence / entity tracking failure, with lexical repetition.

**Observation:** The story starts with three people (the dog, his mom, the doctor), but once "doctor" appears the model uses it for almost every subject and object, which produces sentences like "The doctor told the doctor" and "forgave the doctor". A new character, Lily, also shows up near the end without being introduced. Each sentence is grammatical on its own, so the model has learned local syntax well, but it has no stable record of who is who. A 10.8M-parameter character model has to spend most of its capacity on spelling and word order, and "doctor" is a high-probability continuation after "the" once it has appeared a few times in the context, so the story keeps feeding itself the same noun.

## Failure case 2: hallucination that contradicts the prompt

**Prompt:** "There was a big red ball" (temperature 0.8)

**Snippet:**

```
There was a big red ball. It was blue and had long hair and a swing. It looked like a rainbow.

"Look, a picture," Lily said. She picked it up and hugged it.
```

**Failure type:** hallucination / contradiction of stated facts.

**Observation:** The very next sentence says the red ball is blue and gives it long hair and a swing, and two lines later the object has turned into "a picture" that the rest of the story is about. The model is not keeping track of attributes it has already committed to. In TinyStories, "It was" is usually followed by a color or a description, so the model picks a likely color ("blue") without checking it against "red" a few words earlier. The switch from ball to picture shows the same thing at the object level: the story drifts to whatever noun is most likely next, not the one the prompt set up.

## Failure case 3: repetition and template collapse under greedy decoding

**Prompt:** "Once upon a time" (greedy)

**Snippet:**

```
Lily went to her mom and said, "Mommy, can I have these berries?" Her mom said, "Yes, you can have them." Lily was so happy and thanked her mom. She took a big bite and said, "Mmm, these berries are so yummy!" Her mom smiled and said, "Yes, you can have them."
<EOS>
Once upon a time, there was a little girl named Lily. She
```

The second greedy prompt ends the same way, starting a new story with the identical line "Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine."

**Failure type:** repetition (repeated dialogue line and repeated opening template).

**Observation:** The mother's reply "Yes, you can have them." appears twice, and the second time it doesn't answer anything. After the end-of-story token, both greedy samples fall back to exactly the same opening sentence. Greedy decoding always takes the single most likely next character, and in TinyStories the most common opening by far is "Once upon a time, there was a little girl named Lily", so the model lands on it every time. The metrics show the same pattern: the repeated 4-gram rate is 6.2% for greedy decoding against 0.9% for temperature 0.8 sampling, and distinct-n is higher for the sampled stories. Sampling with temperature 0.8 trades a little coherence for much less repetition, which is why it was used for the main samples.

## What would likely help

A larger context window or a word/subword tokenizer would let the model see more of the story at once and spend less capacity on spelling, which should help with cases 1 and 2. A repetition penalty or top-p sampling would target case 3 directly without having to raise the temperature.
