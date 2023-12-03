# Yahoo-Troll-Questions-Detection
 Given a hand-labeled dataset consisting of questions with unique IDs and whether they are troll questions or not, create a machine learning model to detect spam and troll questions so that they can be removed.

#### Kaggle link: https://www.kaggle.com/competitions/yahoo-troll-question-detection/leaderboard
#### Team: Nimbooz Masala Soda
## Checkpoint 1:

We observed that the data was very skewed towards not-troll. So we came up with an idea,
we decided that we would balance our datasets and then train a classification model on it.

We did the necessary pre-processing:


- removed special characters
- removed numbers
- removed punctuation
- Stemming, etc.

We got a decent F1 accuracy score in our model. But on Kaggle we got very less accuracy.

Inferences Drawn: Later we realized that the testing data itself is biased so balancing the
dataset would prove to be detrimental in this case.

## Checkpoint 2:

Since our original idea of balancing the dataset didn't pan out, we tried a lot of models.

**Model 1 :**


- Did stemming, removed stop words but kept alpha numeric chars
- used BOW + MNB
- got a score of 0. 49

we thought we could do better so we built another model:


**Model 2:**


- Did same pre-processing as above, used BOW+word2vec
- ran a lot of models on this
- MNB - 0. 60
- Random Forest - 0. 5
- Logistic - 0. 64
- Linear SVM - 0. 64

We did a bit of research on similar problem statements in case we found some idea for a
better solution, we found a good research paper regarding the same.

So we did as follows:

**Model 3:**


- Basically removed the whole pre-processing step since we realized that there
- might be a lot of valuable information/ context that were losing.
- We used TF-IDF here.
- Transformed it into naïve Bayes features.
- Did Cross validation using stratified K-fold and logistic regression.
- The most important part here, instead of just straight up doing the binary
- classification of 0/1. We predicted the probability of output being 1.
- The advantage in case of probability here was that later we could manually come
- up with the suitable threshold to classify the output as 0/1.
- We got a pretty good score in this case: 0. 64 (our best yet at that time).

Inferences drawn: Doing pre-processing was causing a lot of information to be lost, thus
resulting in lower accuracy scores, so we directly removed that. Spending a lot of time in
fine tuning the hyper-parameters might pan out and increase our accuracy. We thought at
that time that exploring POS tagging might be a feasible solution in our case.

## Final Model:

We tried out POS tagging finally, we came up with a initial idea that if a question is troll
or not by looking at parts of speech in a question.

We observed that in cases of troll questions, noun and adjective we quite close to each
other. We replaced all words of a ques with their POS tag and then ran our model.


We observed quite low accuracy in this case, as we later realized that our idea itself was a
bit impractical. We then thought if we could combine the POS model and the normal
model for better accuracy, thought we didn't have much time to do that.

At the end, we came up with the following model that ended up with being ranked 2nd on
the private leaderboard :


- We found through trial and error that using NB features worked better than
- when we used BoW/TF-IDF features.
- So, when we used TF-IDF, we transformed it into NB features after that.
- We used the same strategy as above, did stratified K-fold and cross validation.
- We used predict_proba instead of just binary prediction and then did manual
- search for the best threshold.

We spent a lot of time on fine tuning the hyper-parameters and ended up with about 0.64
% accuracy in kaggle.



