import warnings

from pyexpat import features

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , mean_absolute_error
class housepricedetector:
    def __init__(self):
        self.data = None
        self.model = None
        self.features = ['size' , "bedroom" , "age" , 'location']
#will use temporary random data for time being
    def load_sample_data(self):
        np.random.seed(42)
        self.data= pd.DataFrame({
            'size' : np.random.randint(1000,3000,100),
            "bedrooms" : np.random.randint(1,5,100),
            "age" : np.random.randint(1,30,100),
            "location" : np.random.choice([1,2,3],100),
            'price' : np.random.randint(100000,500000,100)
        })
        print('sample data loaded ')
        print(f'dataset shape: {self.data.shape}')
        return self.data
    def explore_data(self):
        if self.data is None:
            print('data not loaded')
            return
        print('data Exploration')
        print('='*40)
        print(self.data.head())
        print("\n data info: ")
        print(self.data.info())
        print("\n basic statistics: ")
        print(self.data.describe())
    def visualize_data(self):
        if self.data is None:
            print('data not loaded')
            return
# price vs size
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2,2 , figsize = (15,12))
        axes[0,0].scatter(self.data['size'], self.data['price'], alpha=0.6, color='blue')
        axes[0,0].set_title('house size vs price ')
        axes[0,0].set_xlabel('size(sq_feet)')
        axes[0,0].set_ylabel('price($)')

#price vs bedroom
        bedroom_groups =[self.data[self.data['bedrooms']==i]['price']for i in range(1,5)]
        axes[0,1].boxplot(bedroom_groups, labels=[1,2,3,4])
        axes[0,1].set_title('price distribution by bedroom')
        axes[0,1].set_xlabel('size(sq_feet)')
        axes[0,1].set_ylabel('price($)')
#price vs age
        axes[1,0].scatter(self.data['age'], self.data['price'], alpha=0.6, color='red')
        axes[1,0].set_title('house age vs price ')
        axes[1,0].set_xlabel('age(years)')
        axes[1,0].set_ylabel('price($)')
#Location vs Price Box Plot
        location_groups = [self.data[self.data['location'] == i]['price'] for i in range(1, 4)]
        axes[1, 1].boxplot(location_groups, labels=['Urban', 'Suburban', 'Rural'])
        axes[1, 1].set_title('Price Distribution by Location')
        axes[1, 1].set_xlabel('Location')
        axes[1, 1].set_ylabel('Price ($)')

        plt.tight_layout()
        plt.savefig('house_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    def build_model(self):
        if self.data is None:
            print("no data loaded")
            return
        print("\n building the model")
        print("=="* 10)
        x= self.data[['size','bedrooms','age','location']]
        y= self.data['price']
        #spliting data into training set
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        print(f'training set : {x_train.shape[0]} houses')
        print(f'testing set : {x_test.shape[0]} houses')
        #crea n train model
        self.model = LinearRegression().fit(x_train, y_train)
        print('the model is trained here ')
        #Make predictions on test set
        y_pred = self.model.predict(x_test)
        #Evaluate model performance
        mae= mean_absolute_error(y_test, y_pred)
        mse= mean_squared_error(y_test, y_pred)
        print('printing model performance')
        print(f"Mean Absolute Error: ${mae:,.2f}")
        print(f"Mean Squared Error: {mse:,.2f}")
        print(f"Average Prediction Error: ±${mae:,.0f}")
        #Show model coefficients (what the model learned)
        print(f"\nWHAT THE MODEL LEARNED:")
        print(f"Base Price: ${self.model.intercept_:,.2f}")
        for i, feature in enumerate(['Size', 'Bedrooms', 'Age', 'Location']):
            print(f"{feature}: ${self.model.coef_[i]:.2f} per unit")
            return self.model
    #for predection
    def predict_price(self, size, bedrooms, age, location):
        if self.model is None:
            print("Model not trained yet! Run build_model() first.")
            return

        features = np.array([[size, bedrooms, age, location]])
        predicted_price = self.model.predict(features)[0]

        print(f"\nPRICE PREDICTION:")
        print(f"Size: {size} sq ft | Bedrooms: {bedrooms} | Age: {age} years | Location: {location}")
        print(f"Predicted Price: ${predicted_price:,.2f}")

        return predicted_price


if __name__ == '__main__':
    predictor = housepricedetector()
    predictor.load_sample_data()
    predictor.explore_data()
    predictor.visualize_data()
    predictor.build_model()

    #Test predictions
    print("\nTESTING PREDICTIONS:")
    predictor.predict_price(2000, 3, 10, 2)
    predictor.predict_price(1500, 2, 5, 1)
    predictor.predict_price(2500, 4, 20, 3)