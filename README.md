# NLP-Spam-Classifier

The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam. This python project of detecting Spam messages deals with spam and ham messages. Using sklearn, we build a TfidfVectorizer on our dataset. Then, we initialize a Naive bayes classifier and fit the model. <br>

We took a [dataset](https://archive.ics.uci.edu/ml/index.php), implemented a TfidfVectorizer, initialized a Naive bayes classifier, and fit our model.
So with this model, we have 955 true positives, 129 true negatives, 0 false positives, and 31 false negatives.We ended up obtaining an accuracy of 97% in magnitude.

Credits: Dataset has bee taken from UCI Machine Learning Repository.Thanks to Krish Naik for the steps and guidance for this beginner NLP model.
