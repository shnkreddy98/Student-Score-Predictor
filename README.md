{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Simple Linear Regression "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This Notebook is part of a series of task given by <a href='https://www.thesparksfoundationsingapore.org/'>The Sparks Foundation</a> as part of GRIP Programme."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Prediction using Supervised Learning "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###### Outcome "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Predict the Percentage of a Student based on the no. of study hours"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [],
   "source": [
    "#importing important libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###### Data Given"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2.5</td>\n",
       "      <td>21</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5.1</td>\n",
       "      <td>47</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.2</td>\n",
       "      <td>27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8.5</td>\n",
       "      <td>75</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.5</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Hours  Scores\n",
       "0    2.5      21\n",
       "1    5.1      47\n",
       "2    3.2      27\n",
       "3    8.5      75\n",
       "4    3.5      30"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "url = 'https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'\n",
    "data = pd.read_csv(url)\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There are only 2 variables in the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(25, 2)"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The dataset has only 25 records, this dataset is comparatively a small dataset."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###### Data Visualization"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Before we apply any any model on some data we need to check the relationship between the data and there is no other good way to check in data than scatter plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.title('Hours vs Scores', fontdict = {'Size' : 15, 'Color': 'Red'})\n",
    "\n",
    "plt.scatter(x = 'Hours', y = 'Scores', data = data)\n",
    "\n",
    "plt.xlabel('No. of Hours Studied', fontdict = {'Size' : 12, 'Color': 'Blue'})\n",
    "plt.ylabel('Marks Scored', fontdict = {'Size' : 12, 'Color': 'Blue'})\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###### Inference:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There is a positive linear relationship between the No. of Hours the student has studied and the marks he has scored. The figure suggests that as the no. of hours the student studies increase the marks they scored also increases."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Keeping all the independent variables in X and putting the target variables in y\n",
    "X = data.iloc[:, :-1]\n",
    "y = data.iloc[:,1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "metadata": {},
   "outputs": [],
   "source": [
    "#splitting data in train and test data\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, train_size = 0.8, random_state = 100)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here train_size = 0.8 & test_size = 0.2 implies that the data is to be split into a ratio of 8:2."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 133,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#fitting the linear model with the train data\n",
    "from sklearn.linear_model import LinearRegression\n",
    "linreg = LinearRegression()\n",
    "linreg.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###### Fitting the trained line through the scatter plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.title('Fitting the trained model to the actual data plot', fontdict = {'Size' : 15, 'Color': 'Red'})\n",
    "\n",
    "plt.scatter(X, y)\n",
    "plt.plot(X, linreg.coef_ * X + linreg.intercept_)\n",
    "\n",
    "plt.xlabel('No. of Hours Studied', fontdict = {'Size' : 12, 'Color' : 'Blue'})\n",
    "plt.ylabel('Marks Scored', fontdict = {'Size' : 12, 'Color' : 'Blue'})\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###### Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = linreg.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Actual Scores</th>\n",
       "      <th>Predicted Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>25</td>\n",
       "      <td>28.545123</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>35</td>\n",
       "      <td>39.364112</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>42</td>\n",
       "      <td>34.446390</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>62</td>\n",
       "      <td>60.018545</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>20</td>\n",
       "      <td>16.742590</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Actual Scores  Predicted Scores\n",
       "9              25         28.545123\n",
       "22             35         39.364112\n",
       "13             42         34.446390\n",
       "11             62         60.018545\n",
       "5              20         16.742590"
      ]
     },
     "execution_count": 136,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Comparing Actual and Predicted Scores\n",
    "y_pred_final = pd.DataFrame({'Actual Scores' : y_test, 'Predicted Scores' : y_pred})\n",
    "y_pred_final"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now that the model is ready and can predict the marks scored by the student when the no. of study hours is given, you can test it on your own data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's say we wanted to know the score of a student who has studied 9.25 hrs/day"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The marks score by a student studying 9.2 hr/day is 92.97\n"
     ]
    }
   ],
   "source": [
    "own_data = [[9.25]]\n",
    "own_data_prediction = linreg.predict(own_data)\n",
    "print('The marks score by a student studying 9.2 hr/day is {0}'.format(round(own_data_prediction[0], 2)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###### Evaluating the model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Absolute Error: 4.140342096254413\n"
     ]
    }
   ],
   "source": [
    "from sklearn import metrics  \n",
    "print('Mean Absolute Error:', \n",
    "      metrics.mean_absolute_error(y_test, y_pred)) "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
