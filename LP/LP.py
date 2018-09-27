
import numpy as np

all_data=np.loadtxt('0620_semi_8020.txt', delimiter=',')

data=all_data[:, 0:3]
label=all_data[:,3]

def make_label_two_col(y):
    output_y_two_col=np.zeros((y.shape[0], 2))
    unlabeled=y==-1
    unlabeled=unlabeled[:,np.newaxis]
    
    neg_ones=np.ones((y.shape[0],2))*-1
    two_col_y=np.zeros((y.shape[0],2))
    two_col_y[:,0]=1-y
    two_col_y[:,1]=y
    output_y_two_col=np.where(unlabeled, neg_ones, two_col_y)
    return output_y_two_col

np.set_printoptions(threshold=np.nan)

label_two_col=make_label_two_col(label)

#print (label_two_col)


from sklearn.semi_supervised import LabelPropagation

label_prop_model=LabelPropagation(max_iter=5000, gamma=1000, n_neighbors=7)


#label_prop_model.fit(data, label)
label_prop_model.fit(data, label_two_col)

predictions=label_prop_model.predict(data)

real_label=np.loadtxt('full_data_with_label.txt', delimiter=',')[:,3]

print (np.mean(predictions==real_label))
