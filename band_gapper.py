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
    fname = '%s/%s.pkl' % (path,pname)

    data.to_pickle(fname)
    print('Saved file to ',fname)
    
    return data

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

def filter_data(df,elems,pname,pmin=None,pmax=None,stab=None):
  
  # filter by chemistry
  inds = np.zeros((len(elems),len(df)))
  for i,item in enumerate(elems):
    inds[i,:] = (df['pretty_formula'].str.contains(item))
    
  idx = np.prod(inds,axis=0)
  df = df[idx==1]
  print(len(df))
  
  # filter by property values
  if pmin:
    df = df[df[pname] >= pmin]
  if pmax:
    df = df[df[pname] <= pmax]
  print(len(df))
    
  # filter by stability
  if stab:
    df = df[df['e_above_hull'] <= stab]
    
  print(len(df))
  
  return df

def add_atom_feats(df):
  
  avg_row = []
  avg_col = []
  avg_num = []
  el_neg = []
  at_mass = []
  at_r = []
  io_r = []
  
  # loop through entries
  for index, row in df.iterrows(): 
    
    comp = Composition(row['pretty_formula'])
    elem,fracs = zip(*comp.fractional_composition.items())

    # 0. average row in the periodic table
    try:
      avg_row.append(sum([el.row*fr for (el,fr) in zip(elem,fracs)]))
    except TypeError:
      avg_row.append(float('nan'))
    
    # 1. average column in the periodic table
    try:
      avg_col.append(sum([el.group*fr for (el,fr) in zip(elem,fracs)]))
    except TypeError:
      avg_col.append(float('nan'))
  
    # 2. average atomic number
    try:
      avg_num.append(sum([el.number*fr for (el,fr) in zip(elem,fracs)]))
    except TypeError:
      avg_num.append(float('nan'))
    
    # 3. average electronegativity
    try:
      el_neg.append(sum([el.X*fr for (el,fr) in zip(elem,fracs)]))
    except TypeError:
      el_neg.append(float('nan'))
    
    # 4. average atomic mass
    try:
      at_mass.append(sum([el.data['Atomic mass']*fr for (el,fr) in zip(elem,fracs)]))
    except TypeError:
      at_mass.append(float('nan'))
    
    # 5. average atomic radius
    try:
      at_r.append(sum([el.data['Atomic radius']*fr for (el,fr) in zip(elem,fracs)]))
    except TypeError:
      at_r.append(float('nan'))
    
    # 6. average ionic radius
    try:
      io_r.append(sum([el.average_ionic_radius*fr for (el,fr) in zip(elem,fracs)]))
    except TypeError:
      io_r.append(float('nan'))
      
  df['avg row'] = pd.Series(avg_row, index=df.index)
  df['avg column'] = pd.Series(avg_col, index=df.index)
  df['avg num'] = pd.Series(avg_num, index=df.index)
  df['avg el-neg'] = pd.Series(el_neg, index=df.index)
  df['avg atom mass'] = pd.Series(at_mass, index=df.index)
  df['avg atom radius'] = pd.Series(at_r, index=df.index)
  df['avg ionic radius'] = pd.Series(io_r, index=df.index)
  
  feat_labels = ['avg row','avg column','avg num','avg el-neg',
                 'avg atom mass','avg atom radius','avg ionic radius']
  
  return df,feat_labels

def add_cs_features(df,rdf_flag=False):

  df["composition"] = str_to_composition(df["pretty_formula"]) 
  df["composition_oxid"] = composition_to_oxidcomposition(df["composition"])
  df["structure"] = dict_to_object(df["structure"]) 

  vo = ValenceOrbital()
  df = vo.featurize_dataframe(df,"composition")

  ox = OxidationStates()
  df = ox.featurize_dataframe(df, "composition_oxid")
  
  # structure features
  den = DensityFeatures()
  df = den.featurize_dataframe(df, "structure")
  
  if rdf_flag:
    rdf = RadialDistributionFunction(cutoff=15.0,bin_size=0.2)
    df = rdf.featurize_dataframe(df, "structure") 
  
  return df