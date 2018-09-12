def encode_chem(pretty_formula):
  '''encode chemistry'''
  
  comp = Composition(pretty_formula).fractional_composition
  
  pt = periodic_table
  elem_list = dir(pt.Element)
  a = [s for s in elem_list if "_" not in s]
  v = [0.0]*len(a)
  
  elements = comp.elements
  
  for el in elements:
    v[elem_list.index(el.name)] = comp[el]

  return v