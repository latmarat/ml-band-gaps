# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import Pipeline  

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from pymatgen.io.cif import CifParser
import requests

def digest(url):
  '''Get structure for given url of CIF file'''

  response = requests.get(url)
  data = response.text

  parser = CifParser.from_string(data)
  structure = parser.get_structures()[0]
  print('Successfuly read structure for',structure.composition)
  
  return structure

def train(x,y,nestim=50,split=True,test_frac=0.1):
    '''Train RF model'''

    ranfor = RandomForestRegressor(n_estimators=nestim,
                            oob_score=True, random_state=42, verbose=0)

    # split train and test data
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_frac, random_state=42)
    print('Data was split with %.3f test fraction' % test_frac)

    # fit RF
    ranfor.fit(x_train,y_train)
    print('RF mode fit using %d estimators ' % nestim)

    # use RF to predict
    y_hat_train = ranfor.predict(x_train) 
    y_hat_test = ranfor.predict(x_test) 

    # print errors
    mae_train = np.mean(abs(y_hat_train-y_train))/np.mean(y_train)
    print('Train error: %.3f ' % mae_train)

    mae_test = np.mean(abs(y_hat_test-y_test))/np.mean(y_test)
    print('Test error : %.3f ' % mae_test)

    return ranfor