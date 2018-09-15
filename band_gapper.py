# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import Pipeline  

import pandas as pd
import requests

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from matminer.utils.conversions import str_to_composition
from matminer.utils.conversions import dict_to_object
from matminer.utils.conversions import composition_to_oxidcomposition

from matminer.featurizers.structure import DensityFeatures
from matminer.featurizers.structure import RadialDistributionFunction
from matminer.featurizers.composition import OxidationStates
from matminer.featurizers.composition import ValenceOrbital

from pymatgen.core.composition import Composition
from pymatgen import Structure
from pymatgen.io.cif import CifParser

def digest(url):
	'''Get structure for given url of CIF file'''

	response = requests.get(url)
	data = response.text

	parser = CifParser.from_string(data)
	structure = parser.get_structures()[0]
	print('Successfuly read structure for',structure.composition)

	return structure

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

def filter_data(df,elems,pname,pmin=None,pmax=None,stab=None):
	'''Filter data by criteria'''

	print('# entries before filters: ',len(df))

	# filter by chemistry
	inds = np.zeros((len(elems),len(df)))
	for i,item in enumerate(elems):
	  inds[i,:] = (df['pretty_formula'].str.contains(item))
	  
	idx = np.prod(inds,axis=0)
	df = df[idx==1]
	print('# entries after chemistry: ',len(df))

	# filter by property values
	if pmin:
	  df = df[df[pname] >= pmin]
	if pmax:
	  df = df[df[pname] <= pmax]
	print('# entries after property: ',len(df))
	  
	# filter by stability
	if stab:
	  df = df[df['e_above_hull'] <= stab]
	print('# entries after stability: ',len(df))

	return df

def get_xy(df,elems,pname,pmin,pmax,stab):
	'''Get x and y from data'''

	# filter NaNs and entries based on criteria
	df = df.dropna()
	df = filter_data(df,elems,pname,pmin=pmin,pmax=pmax,stab=stab)

	# exclude non-input columns
	exclude = ['pretty_formula',pname,'e_above_hull','structure','composition','composition_oxid','radial distribution function']
	
	# get X and Y
	x = df.sort_index().drop(exclude, axis=1)
	y = df[pname].sort_index().values

	return x,y

def fit_forest(x,y,lbl='Full'):

  # split data
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  # grid-search optimal parameters
  rf = RandomForestRegressor()
  param_grid = { 
        'n_estimators'      : [10,25,50,100,250],
        'max_features'      : ['auto','sqrt','log2'],
        'min_samples_split' : [2,4,8],
        'min_samples_leaf'  : [1, 2, 5]
        }
  grid = GridSearchCV(rf, param_grid, n_jobs=-1, cv=5)
  grid.fit(x_train, y_train)

  print(grid.best_score_)
  print(grid.best_params_)
  print(grid.score(x_test, y_test))

  # use optimal parameters
  rf.set_params(**grid.best_params_)
  rf.fit(x_train, y_train)

  y_hat_train = rf.predict(x_train) 
  y_hat_test = rf.predict(x_test) 

  mae_train = np.mean(abs(y_hat_train-y_train))/np.mean(y_train)
  print('%s RF, train error: %.3f' % (lbl,mae_train))

  mae_test = np.mean(abs(y_hat_test-y_test))/np.mean(y_test)
  print('%s RF, test error : %.3f' % (lbl,mae_test))

  return rf

def fit_model(x,y,show_flag=False):

  # fit RF using all variables
  print('Fitting full random forest...')
  rf = fit_forest(x,y,lbl='Full')

  # variable importances
  nvar = 10
  imp = rf.feature_importances_
  idx = np.argsort(imp)[::-1]
  print('%d most important variables:' % nvar)
  print(x.columns.values[idx][0:nvar])

  # prune variables
  thr = 0.5*np.median(imp)
  idx = imp < thr
  exclude = list(x.columns.values[idx])
  x_sel = x.drop(exclude, axis=1)

  # fit RF using important variables
  print('\nFitting pruned random forest...')
  rf = fit_forest(x_sel,y,lbl='Pruned')
  
  print('%d pruned variables:' % len(x_sel.columns))
  print(x_sel.columns.values)
  
  if show_flag:
    # plt.figure(figsize=(7, 4))

    # importance chart
    plt.subplot(121)
    
    ind = np.argsort(imp)[::-1]
    plt.bar(x=x.columns.values[ind][0:nvar], height=imp[ind][0:nvar],color=(0.3,0.3,0.9))
    plt.xticks(x.columns.values[ind][0:nvar], x.columns.values[ind][0:nvar], rotation='vertical')
    plt.xlabel('Variables')
    plt.ylabel('Importance')

    # parity plot
    ax = plt.subplot(122)
    ax.set_aspect(1)
    
    plt.scatter(y, rf.predict(x_sel),marker='s',alpha=.25,c=(0.9,0.3,0.3))
    plt.plot(np.arange(np.max(y)),c='gray')
    plt.xlabel('Ground truth')
    plt.ylabel('RF prediction')
    
    plt.subplots_adjust(bottom=0.25,top=0.75)
    plt.draw()
    plt.show()

  return rf

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