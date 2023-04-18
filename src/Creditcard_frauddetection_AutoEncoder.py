#!/usr/bin/env python
# coding: utf-8

# Yuxin(Yuki) Chen          
# Siyue(Sherry) Gao

# ### Import packages and data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.over_sampling import BorderlineSMOTE
import seaborn as sns
import plotly.express as px
warnings.filterwarnings('ignore')

import tensorflow as tf
import plotly.graph_objs as go
import plotly.offline as pyo

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, average_precision_score, roc_curve, auc


# In[2]:


df = pd.read_csv("data/creditcard.csv")


# In[3]:


df.head()


# ### Data preprocessing

# In[4]:


print(df.describe())


# In[5]:


#no null data
null_col = df.isnull().sum()
print(null_col)


# #### Data scaling

# In[6]:


scaler = MinMaxScaler(feature_range = (-1, 1))
scaled_df = scaler.fit_transform(df.iloc[:,1:30])
df.iloc[:,1:30] = scaled_df


# In[7]:


print(df.describe())


# ### Feature engineering

# In[8]:


sns.distplot(df[df['Class']==1]['Time'], hist=False, label='Class 1')
sns.distplot(df[df['Class']==0]['Time'], hist=False, label='Class 0')
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Distribution of time in two classes')
plt.show()


# In[9]:


#feature selection
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=18)
df_rfe = rfe.fit_transform(df, df['Class'])


# In[10]:


selected_features = rfe.support_
print(df.columns[selected_features])


# In[11]:


new_df = df[df.columns[selected_features]]


# In[12]:


new_df.head()


# ### Model trainning

# In[13]:


new_df['Class'].value_counts().plot(kind='bar')
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Category Distribution")
for k, v in enumerate(list(new_df.groupby("Class").count().iloc[:,1])):
    plt.text(k, v+0.1, str(v), ha='center')
plt.show()


# #### Data enhancement

# ##### Oversampling

# In[14]:


resampler = BorderlineSMOTE(random_state = 511, kind='borderline-1')
X_re_over, y_re_over = resampler.fit_resample(new_df.iloc[:,:-1], new_df.iloc[:,-1])


# In[15]:


y_re_over.value_counts().plot(kind='bar')
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Resample Category Distribution")
for k, v in enumerate(list(y_re_over.value_counts())):
    plt.text(k, v+0.1, str(v), ha='center')
plt.show()


# #### Model

# #### dataset

# In[16]:


#train, test
X_train, X_test, y_train, y_test  = train_test_split(X_re_over, y_re_over, test_size=0.3,random_state=511)


# #### AE_original

# In[17]:


input_shape = X_train.shape[1]


# In[18]:


class AE_original(tf.keras.Model):
    def __init__(self, latent_view_dim):
        super(AE_original, self).__init__()
        self.latent_view_dim = latent_view_dim
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_shape, ), name='input'),
            tf.keras.layers.Dense(15,
                                  activation='relu',
                                  name='encoder1'),
            tf.keras.layers.Dense(latent_view_dim,
                                  activation='tanh',
                                  name='latent')
        ],
                                           name='Encoder')

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_view_dim, )),
            tf.keras.layers.Dense(15,
                                  activation='relu',
                                  name='decoder1'),
            tf.keras.layers.Dense(input_shape, name='output')
        ],
                                           name='Decoder')
        self.latent_view = self.encoder.layers[-1].output
        self.AE_model = tf.keras.Model(inputs=self.encoder.input,
                                       outputs=self.decoder(self.latent_view),
                                       name='AutoEncoder')

    def call(self, input_tensor):
        reconstruction = self.decoder(self.encoder(input_tensor))
        return reconstruction

    def summary(self):
        return self.AE_model.summary()

    def save_model(self, filepath):
        self.AE_model.save(filepath)


# In[19]:


seed = 95
tf.random.set_seed(seed)
np.random.seed(seed)
model_ori = AE_original(10)
model_ori.summary()


# In[20]:


model_ori.compile(optimizer = 'adam', loss = 'mse', metrics=['mse','acc'], experimental_run_tf_function = False)
X_train_norm = X_train[y_train == 0].values
X_test_norm = X_test[y_test == 0].values
history = model_ori.fit(X_train_norm, X_train_norm, 
              validation_data = (X_test_norm, X_test_norm), 
              batch_size = 128, epochs = 10, verbose = 2)


# In[37]:


plt.figure()
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Original AE train&val loss')
plt.legend()
plt.show()


# In[22]:


model_ori.save_model('model/AE_model_original')


# #### AE_improved

# In[30]:


input_shape = X_train.shape[1]


# In[31]:


class AE(tf.keras.Model):
    def __init__(self, latent_view_dim):
        super(AE, self).__init__()
        self.latent_view_dim = latent_view_dim
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_shape, ), name='InputLayer'),
            tf.keras.layers.Dense(200,
                                  kernel_initializer='uniform',
                                  activation='tanh',
                                  name='encoder1'),
            tf.keras.layers.Dense(150,
                                  kernel_initializer='uniform',
                                  activation='tanh',
                                  name='encoder2'),
            tf.keras.layers.Dense(100,
                                  kernel_initializer='uniform',
                                  activation='tanh',
                                  name='encoder3'),
            tf.keras.layers.Dense(50,
                                  kernel_initializer='uniform',
                                  activation='tanh',
                                  name='encoder4'),
            tf.keras.layers.Dense(latent_view_dim,
                                  kernel_initializer='uniform',
                                  activation='tanh',
                                  name='latent')
        ],
                                           name='Encoder')

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_view_dim, )),
            tf.keras.layers.Dense(50,
                                  kernel_initializer='uniform',
                                  activation='tanh',
                                  name='decoder1'),
            tf.keras.layers.Dense(100,
                                  kernel_initializer='uniform',
                                  activation='tanh',
                                  name='decoder2'),
            tf.keras.layers.Dense(150,
                                  kernel_initializer='uniform',
                                  activation='tanh',
                                  name='decoder3'),
            tf.keras.layers.Dense(200,
                                  kernel_initializer='uniform',
                                  activation='tanh',
                                  name='decoder4'),
            tf.keras.layers.Dense(input_shape, kernel_initializer='uniform', name='output')
        ],
                                           name='Decoder')
        self.latent_view = self.encoder.layers[-1].output
        self.AE_model = tf.keras.Model(inputs=self.encoder.input,
                                       outputs=self.decoder(self.latent_view),
                                       name='AutoEncoder')

    def call(self, input_tensor):
        reconstruction = self.decoder(self.encoder(input_tensor))
        return reconstruction

    def summary(self):
        return self.AE_model.summary()

    def save_model(self, filepath):
        self.AE_model.save(filepath)


# In[32]:


seed = 95
tf.random.set_seed(seed)
np.random.seed(seed)
model = AE(100)
model.summary()


# In[33]:


model.compile(optimizer = 'adam', loss = 'mse', metrics=['mse','acc'], experimental_run_tf_function = False)
X_train_norm = X_train[y_train == 0].values
X_test_norm = X_test[y_test == 0].values
h = model.fit(X_train_norm, X_train_norm, 
              validation_data = (X_test_norm, X_test_norm), 
              batch_size = 256, epochs = 10, verbose = 2)


# In[36]:


plt.figure()
plt.plot(h.history['loss'], label = 'loss')
plt.plot(h.history['val_loss'], label = 'val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Improved AE train&val loss")
plt.legend()
plt.show()


# In[35]:


model.save_model('model/AE_model_imporved')


# ### Evaluation

# In[23]:


def AE_viz(train_result):
    precision, recall, threshold = precision_recall_curve(train_result['true'], train_result['pre'])
    f1_score = 2 * precision * recall / (precision + recall)
    average_precision = average_precision_score(train_result['true'], train_result['pre'])
    print("average preciseion: ", average_precision)
    max_f1 = f1_score[f1_score == max(f1_score)]
    best_threshold = threshold[f1_score[1: ] == max_f1]
    
    # Precision-Recall Curve
    trace1 = go.Scatter(x=threshold, y=precision[1:], name="Precision", line=dict(width=3))
    trace2 = go.Scatter(x=threshold, y=recall[1:], name="Recall", line=dict(width=3, color='red'))
    layout1 = go.Layout(title="Precision and Recall for Different Threshold Values", 
                    xaxis=dict(title="Threshold"), yaxis=dict(title="Precision&Recall"), 
                    hovermode='closest', legend=dict(x=0.75, y=1))
    fig1 = go.Figure(data=[trace1, trace2], layout=layout1)
    
    # F1 Score Curve
    trace5 = go.Scatter(x=threshold, y=f1_score[1:], name="F1-score Curve", line=dict(width=3, color='purple'))
    layout2 = go.Layout(title="F1 Score for Different Threshold Values", 
                    xaxis=dict(title="Threshold"), yaxis=dict(title="F1 Score"), 
                    hovermode='closest', legend=dict(x=0.75, y=1))
    fig2 = go.Figure(data=[trace5], layout=layout2)

    pyo.init_notebook_mode(connected=True)
    pyo.iplot(fig1)
    pyo.iplot(fig2)
    
    # Best Threshold and Max F1 Score
    print('Best threshold = %f' % (best_threshold))
    print('Max F1 score = %f' % (max_f1))
    
    return best_threshold


# In[24]:


#AE predictor & viz
def AE_predictor_viz(X_test_true,y_test,model, threshold):
    X_test = model.predict(X_test_true)
    mse = np.mean(np.power(X_test - X_test_true, 2), axis=1)
    y = np.zeros(shape=mse.shape)
    y[mse > threshold] = 1

    cm = confusion_matrix(y_test.values, y)
    print(classification_report(y_test.values, y))

    fig = px.imshow(cm,
                    color_continuous_scale='Blues',
                    x=['0', '1'],
                    y=['0', '1'],
                    text_auto=True)
    fig.update_layout(
        title={
            'text': "Confusion Matrix",
            'x': 0.5,
            'y': 0.95,
            'font': dict(size=20),
            'xanchor': 'center',
            'yanchor': 'top'
        })
    fig.show()

    return y


# In[25]:


def roc_viz(train_result):
    fpr, tpr, thresholds = roc_curve(train_result['true'], train_result['pre'])
    roc_auc = auc(fpr, tpr)
    #ROC curve
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=fpr,
                   y=tpr,
                   mode='lines',
                   name='ROC Curve (AUC = %0.3f)' % roc_auc))
    fig.add_trace(
        go.Scatter(x=[0, 1],
                   y=[0, 1],
                   mode='lines',
                   line=dict(dash='dash', color='gray'),
                   showlegend=False))
    fig.update_layout(title='Receiver operating characteristic curve (ROC)',
                      xaxis_title='False Positive Rate (FPR)',
                      yaxis_title='True Positive Rate (TPR)',
                      xaxis_range=[-0.02, 1],
                      yaxis_range=[0, 1.02])
    fig.show()


# #### AE_original

# In[26]:


train_pred = model_ori.predict(X_train.values)
train_mse = np.mean(np.power(train_pred - X_train.values, 2), axis=1)
train_result = pd.DataFrame({'pre':train_mse,'true':y_train.values})


# In[27]:


best_threshold = AE_viz(train_result)


# In[28]:


y_pred = AE_predictor_viz(X_test_true = X_test.values,y_test = y_test, model = model_ori, threshold = best_threshold)


# In[29]:


roc_viz(train_result)


# #### AE_improved

# In[38]:


train_pred = model.predict(X_train.values)
train_mse = np.mean(np.power(train_pred - X_train.values, 2), axis=1)
train_result = pd.DataFrame({'pre':train_mse,'true':y_train.values})


# In[39]:


best_threshold = AE_viz(train_result)


# In[40]:


y_pred = AE_predictor_viz(X_test_true = X_test.values,y_test = y_test, model = model, threshold = best_threshold)


# In[41]:


roc_viz(train_result)

