# -*- coding: utf-8 -*-


from   sklearn.neural_network    import MLPClassifier
from   sklearn.model_selection   import cross_validate
from   sklearn.datasets          import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np



# Loading a dataset
Cancer_data = load_breast_cancer()

# Create classifier object model
# Επιλέγω κάθε φορά hidden layers στο hidden_layer_sizes=(?, ?)
print ("Classifier: MLPClassifier\n")
print(Cancer_data.filename)

hid_lay=[(4,2),(8,4),(4,4)]

Accuracy= []
F1= []

for j in range (0,3):
    classifier = MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto', 
                               beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
                               hidden_layer_sizes=hid_lay[j], learning_rate='constant',
                               learning_rate_init=0.001, max_iter=1000, momentum=0.2,
                               nesterovs_momentum=True, power_t=0.5, random_state=0,
                               shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.1,
                               verbose=False, warm_start=False)


    # Print number of instances and attributes
    print ("\n")
    print("Number of instances:  ", len(Cancer_data.data))
    print ("Number of attributes: ", len(Cancer_data.feature_names))
    print ("\n")
    print ("Hidden_layers: ", classifier.hidden_layer_sizes)
    print ("\n")

    # Define inputs and targets
    X = Cancer_data.data
    Y = Cancer_data.target


    # Performance metrics
    scoring_metrics = ['f1', 'accuracy']

    # k-Cross validation 
    scores = cross_validate(classifier, X, Y, scoring=scoring_metrics, cv=10, 
                       return_train_score=False)



    # Print experimental analysis
    print ('Fold   Accuracy       F1')
    for i in range(10):
        print ('  %2i     %.2f%%   %.2f%%' 
               % (i+1, 100*scores['test_accuracy'][i], 100*scores['test_f1'][i]))
    
    print ('---------------------------------------------------------')
    print ('Average: %.2f%%   %.2f%%' 
           % (100*scores['test_accuracy'].mean(), 100*scores['test_f1'].mean()))

        
    A_md= round(100*scores['test_accuracy'].mean(),2)
    F_md= round(100*scores['test_f1'].mean(),2)
    
    Accuracy.append(A_md)
    F1.append(F_md)

print ("\n")
print(Accuracy)
print(F1)

n_groups = 3

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, Accuracy, bar_width, alpha=opacity, color='b', label='Accuracy')
rects2 = plt.bar(index + bar_width, F1, bar_width, alpha=opacity, color='g', label='F1')

plt.yticks([0, 20, 40, 60, 80, 100])
plt.xlabel('Accuracy - F1')
plt.ylabel('Percent')
plt.title('Accuracy - F1')
plt.xticks(index + bar_width/2, ('HL: 4,2', 'HL: 8,4', 'HL: 4,4'))
plt.legend()

plt.tight_layout()
plt.show()