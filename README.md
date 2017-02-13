# Enron Email Investigation

In the wake of the Enron scandal, a lot of people wondered why and how such a large illegitimate operation was going on unnoticed. With the help of modern machine learning models, we can retroactively attempt to piece together the puzzle all while gaining valuable insight on how we might use similar models for auditing purposes in the future.

## Dataset

The goal of this project is to find person's of interest (POIs) in the Enron corporation. To accomplish this, we are looking at a large collection of emails as their respective employee records. In total, there are 146 employees on record, of which only 11 of them have been identified as POIs. The low frequency of POIs in this dataset will present a challenge for the classifiers depending on the features available. This task is made even harder still by the lack of data for a lot of the 20 total features resulting in a fairly sparse data set. These factors combined with just low amount of data initially available make creating a robust model difficult to create.

## Outliers in dataset

Upon initial exploration of the dataset, I came across two very strange outliers that needed to be removed from the dataset. One was a simple aggregation key named "TOTAL" which is probably a remnant from a copy-paste from a spreadsheet. This aggregate key was quite noticeable in regression plots of continuous features and thus easily removed from consideration. Additionally, there was an unexplained key value that had the name "THE TRAVEL AGENCY IN THE PARK". Unlike the rest of the records, this clearly wasn't referring to a person. Also, its corresponding entries were fairly sparse which justified my decision to remove it from the dataset. Once these two outliers were removed, there still remained several, less-severe outliers but they actually seemed to hold some significance for this investigation given that these outliers corresponded to high ranking executives in the company.

## Features

As previously mentioned, there are 20 features in the dataset corresponding to 2 main broad categories: financial features and email features. Like the names imply, financial features captures various financial aspects of Enron employees including salary, bonuses, stock options, etc. whereas the other main group of features revolve around email aggregate statistics such as emails sent, received, recipients that are POIs, etc. As part of my exploration, I constructed an additional feature that highlights the bivariate relationship between bonus and salary. The reasoning is that a higher bonus coupled with a low salary might signify some shady transactions since you would normally expect a bonus received to be roughly proportional to the employee's salary. A high bonus might mean they were getting paid off by someone and could be an indirect sign of corruption within the company.

After creating the additional bonus to salary ratio feature, I preprocessed the data to reduce its dimensionality in order to reduce variance to the data. The method I used was principal component analysis which is great for reducing highly dimensional data by reducing features to projections that cover the most variance in the dataset per "feature". From the 21 total features, I was able to create 11 principal components for my final model. For the algorithms I would eventually apply, feature scaling would not be appropriate. Naive Bayes and Logistic Regression do not respond algorithmically to scaling in the data and while it does matter for RBF variants of SVMs, the linear one I use does not require it.

## Algorithm Selection and Tuning

To create the model, I initially started with three different algorithms: Naive Bayes (Gaussian), Logistic Regression, and a Linear SVM. In order to tune these algorithms, I used Sklearn's `GridSearchCV()` function which allowed me to tune the hyper parameters of these algorithms automatically with a stratified k-fold cross validation where k = 3. Surprisingly however, the algorithms showed very little change with "optimal" configurations compared to default runs.

In fact, most of the variance in results came through manipulation of the PCA components. Performance was low for each algorithm without PCA as most of them would be in the low 20s for precision and recall. With PCA reducing the number of components to 11, the best *consistent* result was obtained from the Gaussian Naive Bayes which got an F1 score of 0.36337, Precision of .44198, and Recall of .30850.

In the context of this investigation, the distinction between the evaluation metrics is important for contextualizing the performance of the algorithm. Recall gives us our ratio of correctly identifying POIs from the whole set of POIs. Precision in this case demonstrates how discriminant our algorithm was in identifying POIs i.e. for all people identified as POIs, what ratio of them were actually POIs? F1 score is a useful "average" of these two and was the basis for my comparisons of performance among the 3 chosen algorithms.

Given that the highest performing algorithm was the Naive Bayes classifier, there was not really an opportunity to further tune it as the only real way to would be to incorporate priors into which is not known unfortunately. Since there was not much to be done with regards to improving the Naive Bayes approach, I was able to tune the remaining algorithms using the aforementioned strategy of k-fold cross validation in the `GridSearchCV` function. This step is crucial for assessing performance as it serves as a check on "overtuning" these algorithms to the training data. This would negatively impact its ability to generalize to unseen information. Various methods of cross validation alleviate this issue by cleverly increasing the amount of testing data used in total to train the algorithm by splitting it into k-folds (in my case, 3 folds!)

## Conclusion

Overall, identifying POIs in this case has proven to be quite the challenge. As evidenced by the low relative F1 scores of each algorithm even after being tuned, this suggests that the combination of features utilized proved to not be very effective. I further verified this by testing the performance of the algorithms with and without my added feature and there was very little difference. There are two solutions which come to mind in order to improve performance, one is to simply aquire more data. This dataset is fairly small relative to the dimensionality of the feature space. PCA in this situation was definitely needed in order to boost performance but even that can only go so far.  

### References

* Udacity Intro to Machine Learning
* [Sklearn Documentation](http://scikit-learn.org/stable/documentation.html)
