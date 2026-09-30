# Task 2 error review: 20 errors from my best model (Exp 2, BiLSTM)

Model: `checkpoints/exp2_bilstm_best.pt` (93.09% test accuracy). The 20 errors below were selected automatically from the 38,000-review test set by `src/sentiment.py` and are listed in `outputs/exp2_bilstm/error_review.csv` (review text shortened there to 1,200 characters): the 5 most confident false positives, the 5 most confident false negatives, the 5 errors closest to the 0.5 threshold, and 5 errors from the weakest slice (reviews containing a negation word). Label 1 = positive, 0 = negative. In Yelp Polarity, 1–2 star reviews are negative and 3–4 star reviews are positive, so the label comes from the star rating, not from the text.

## Summary of error types

| error type | count | reviews |
|---|---|---|
| Mixed or contrastive review (praise and complaint together; the verdict hinges on "but", "not great", or a comparison) | 7 | 408, 10770, 25701, 34570, 11853, 17022, 14750 |
| Positive overall verdict buried under complaints, often at the very end | 4 | 22451, 11808, 11699, 29196 |
| Label does not match the text (edited review or star rating that contradicts the words) | 2 | 29330, 22807 |
| Negative words aimed at someone other than the business | 1 | 21215 |
| Temporal contrast ("used to be bad, now good") | 1 | 34219 |
| Sarcasm, idioms or figurative language | 2 | 2362, 7870 |
| Little or no sentiment signal | 2 | 1404, 888 |
| Non-English review | 1 | 19216 |

The biggest group by far is reviews where praise and complaint appear together and the label depends on how they are weighed. Several of these turn on exactly the words my preprocessing throws away (see the testable fix below).

## The 20 errors

### Confident false positives (true: negative, predicted: positive)

| # | test index | p(positive) | error type | what happened | fix |
|---|---|---|---|---|---|
| 1 | 408 | 0.999 | Mixed / comparison | A 2-star order from Maharani full of positive words ("very quickly", "pretty good", "amazing", "great"). The negative verdict is only implied by comparing it to another restaurant ("Copper is definitely still my place, but Maharani was fine enough"). | Keep contrast words so "but ... fine enough" survives preprocessing |
| 2 | 10770 | 0.999 | Mixed (quality vs value) | Every dish is praised ("perfectly cooked", "excellent", "great"), and the complaint is about portion size and price. The model counts the praise. | Keep contrast words; aspect-level signals would help (value vs quality) |
| 3 | 29330 | 0.999 | Label noise | The text is entirely positive ("love the place", "very clean and new", "great place ... worth a try") but the rating was 1–2 stars. No text model can get this right. | None from the model side; flag likely mislabels by checking high-confidence disagreements |
| 4 | 11699 | 0.999 | Verdict late in a long review | A long, friendly story about a cupcake shop ("absolutely gorgeous") with the actual disappointment (too much frosting, dense cake) late in the review. | Attention pooling, so the model can weight the sentences that carry the verdict |
| 5 | 25701 | 0.998 | Contrastive, faint praise | "Filling and good.....but not great ... there are much better options". The positive adjectives outnumber the one qualifying phrase. | Keep contrast words ("but") |

### Confident false negatives (true: positive, predicted: negative)

| # | test index | p(positive) | error type | what happened | fix |
|---|---|---|---|---|---|
| 6 | 22807 | 0.0003 | Label noise (edited review) | The review starts "EDIT: ... Horrible service" and is negative throughout, but the positive star rating is left over from before the edit. | None from the model side; edited reviews are a known source of Yelp label noise |
| 7 | 22451 | 0.0006 | Verdict at the end | Almost the whole review attacks the food ("crap", "horrible", "worst nachos"), and only the last lines say Thursday nights keep a four-star rating. | Attention pooling to weight the summary sentences |
| 8 | 11808 | 0.0007 | Verdict buried under humorous complaints | Complaints dominate ("TINY", "overpriced", "stupid expensive") but the tone is affectionate and the overall rating positive ("totally the highlight", "points for that"). | Hard for this model; would need more context or an attention layer to weight the summary sentences |
| 9 | 21215 | 0.0007 | Sentiment aimed at a third party | The negative words ("rude", "swearing", "offended") describe other customers; the business and its bartender are praised ("very cordial ... the right thing"). | Hard without knowing the target of each sentiment; aspect/target-aware models address this |
| 10 | 34219 | 0.0013 | Temporal contrast | "Before food was bland and took long ... however I have been impressed ... food quality has improved". The negative past is spelled out, the positive present less so, and "however" is removed as a stopword. | Keep contrast words ("however") |

### Near-threshold errors (true: negative, p just above 0.5)

| # | test index | p(positive) | error type | what happened | fix |
|---|---|---|---|---|---|
| 11 | 888 | 0.5001 | Weak domain signal, long | A rant about a doctor's suit and business model. Most of it is questions and opinions about attire, with few of the restaurant-style sentiment words the model learned. | More non-restaurant training data |
| 12 | 34570 | 0.5001 | Contrastive opening | Opens with "I love vermicelli but" and then lists problems ("watered down", "tasted bad", "no ... flavor"). "Love" pulls one way, the complaints the other. | Keep contrast words |
| 13 | 11853 | 0.5003 | Neutral, lukewarm review | "Not earth-shattering, but not bad either". The text is genuinely close to neutral, so a probability of 0.5 is a fair answer. | Calibrated abstention: send reviews with p near 0.5 for review instead of forcing a label |
| 14 | 19216 | 0.5003 | Non-English (French) | Preprocessing keeps only a–z, so accented French words are broken into fragments ("d sagr able") that are mostly unknown to the vocabulary. The model has almost nothing to go on. | Keep accented letters, or detect the language and handle non-English reviews separately |
| 15 | 1404 | 0.5006 | Almost no sentiment words | "CLOSED! ... Dust bunnies, anyone?" The review is sarcastic and very short. | Hard for any bag of short signals; would need a model that understands sarcasm |

### Errors in the weakest slice (reviews with a negation word)

| # | test index | p(positive) | error type | what happened | fix |
|---|---|---|---|---|---|
| 16 | 2362 | 0.817 | Figurative language | "The more beautiful, delicious cousin of Rubio's" is nostalgia, and "tasted fake, like a plastic doll" is a metaphor. The positive words are literal, the complaints are figurative. | Hard; more data with figurative complaints |
| 17 | 7870 | 0.666 | Idioms, out-of-domain | A casino review: "no win situation", "money hungry man eaters". Few food words and idiomatic negativity. | More non-restaurant data |
| 18 | 17022 | 0.607 | Comparison to other businesses | Clearly negative ("less than welcoming", "bad service", "we will not be returning"), but ends by praising *other* restaurants ("great service and great food"). | Keep contrast/comparison words; weight the end of the review |
| 19 | 14750 | 0.834 | Negation scope and hedged praise | "I am not proud to admit this ... Don't get me wrong the place is kept clean". Negations here soften the praise rather than flip it, and the complaint ("so greasy") is short. | Keep hedge words ("but", "only"); longer training on negation patterns |
| 20 | 29196 | 0.085 | Many small negatives inside a positive review | A long item-by-item review with an overall positive tone ("Amazing!", "Delicious, best item") but many local negatives ("didn't particularly care", "I don't like pickled veggies", "okay"). Max pooling picks up the strongest local signals, and here the local signals are mostly negative. | Attention pooling, weighting overall statements over item-level remarks |

## One testable fix: stop removing contrast and intensity words

While going through these errors I checked which words the scikit-learn stopword list removes. Negations are kept on purpose (`keep_negations = 1`), but the list also removes **"but", "however", "although", "though", "yet", "too", "very", "less", "only", "still", "even", "except", "enough", "rather"**. These are exactly the words that decide mixed reviews: "good but not great", "fine enough", "however I have been impressed", "too much frosting". At least 7 of the 20 errors above (408, 10770, 25701, 34570, 34219, 17022, 14750) turn on one of them.

**Hypothesis.** Keeping these words will reduce errors on mixed and contrastive reviews.

**Test.**
1. Add the words above to the set of stopwords that are kept, alongside the negations.
2. Retrain the same BiLSTM with the same seed and hyperparameters.
3. Compare against the current model on the same test set: macro-F1 with its bootstrap confidence interval, a McNemar test between the two models, and the error rate on a new slice of test reviews that contain any of these words.

**Success criterion.** A lower error rate on the contrast-word slice and a McNemar p-value below 0.05, without a drop in overall macro-F1.

A second fix worth testing the same way is **attention pooling** instead of max pooling over the BiLSTM outputs, which targets errors 4, 7 and 20, where the verdict sits in a few sentences of a long review. Truncation is not the cause here: after cleaning, only 1.4% of reviews exceed the 256-token limit. Its success criterion would be a lower error rate on the "long (> 142 words)" slice, currently 7.1%.

## Label noise

Errors 3 and 6 are not model mistakes: the text says one thing and the star rating says the other. Both got near-certain wrong predictions (0.999 and 0.0003), which is typical of mislabelled examples. Some of the model's "confident errors" are therefore label noise in the dataset rather than model failures, and the true accuracy is slightly higher than the measured 93.09%.
