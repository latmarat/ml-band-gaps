# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import Pipeline  

import pandas as pd
import requests

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from pymatgen.io.cif import CifParser

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
    print('Split data with %.3f test fraction' % test_frac)

    # fit RF
    ranfor.fit(x_train,y_train)
    print('Fit RF model using %d estimators ' % nestim)

    # use RF to predict
    y_hat_train = ranfor.predict(x_train) 
    y_hat_test = ranfor.predict(x_test) 

    # print errors
    mae_train = np.mean(abs(y_hat_train-y_train))/np.mean(y_train)
    print('Train error: %.3f ' % mae_train)

    mae_test = np.mean(abs(y_hat_test-y_test))/np.mean(y_test)
    print('Test error : %.3f ' % mae_test)

    return ranfor

def predict(x_new,model):
    '''Predict using trained model'''

    y_new = model.predict(x_new)

    return y_new

def use(formula,url):
    '''Use model for new structures'''

    # get structure from CIF
    s = digest(url)

    atom_feats = get_atom_feats(formula)
    lat_feats = get_crys_feats(formula)
    rdf = radial_dist_func(s)

    # initialize dataframe
    crys_data = pd.DataFrame(columns=['pretty_formula','initial_structure',
                  'atomic_features','lattice_features','radial_dist','chem'])

    crys_data["composition"] = str_to_composition(crys_data["pretty_formula"]) 
    cp.ValenceOrbital().featurize_dataframe(crys_data, col_id="composition")  

    return y_new


def query_data(pname,api_key,path=''):


    mpdr = MPDataRetrieval(api_key)

    # query properties
    props = mpdr.get_dataframe(criteria={pname: {"$exists": True},
#                                      "elements": {"$all": ["Li", "Fe", "O"]},
                                       ("{}.warnings".format(pname)): None},
                             properties=['pretty_formula',pname,'e_above_hull'])
    print("There are {} entries satisfying criteria".format(props[pname].count()))

    # Load crystal structures
    # initialize dataframe
    structures = pd.DataFrame(columns=['structure'])

    # lists of mp ids to avo
    chunk_size = 1000
    mp_ids = props.index.tolist()
    sublists = [mp_ids[i:i+chunk_size] for i in range(0, len(mp_ids), chunk_size)]

    # query structures 
    for sublist in sublists:
        structures = structures.append(mpdr.get_dataframe({"material_id":{"$in": sublist}}, ['structure']))
    
    data = pd.concat([props,structures],axis=1)
    fname = '%s/%s.csv' % (path,pname)

    data.to_csv(fname)
    print('Saved file to ',fname)
    
    return data