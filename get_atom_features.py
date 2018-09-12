def get_atom_feats(pretty_formula):

  comp = Composition(pretty_formula)
  elem,fracs = zip(*comp.fractional_composition.items())

  v = []

  # 0. average row in the periodic table
  v.append(sum([el.row*fr for (el,fr) in zip(elem,fracs)]))
  
  # 1. average column in the periodic table
  v.append(sum([el.group*fr for (el,fr) in zip(elem,fracs)]))

  # 2. average atomic number
  v.append(sum([el.number*fr for (el,fr) in zip(elem,fracs)]))

  # 3. average electronegativity
  v.append(sum([el.X*fr for (el,fr) in zip(elem,fracs)]))
    
  # 4. average atomic mass
  v.append(sum([el.data['Atomic mass']*fr for (el,fr) in zip(elem,fracs)]))

  # 5. average atomic radius
  v.append(sum([el.data['Atomic radius']*fr for (el,fr) in zip(elem,fracs)]))

   
  # ------------------------------------------------------------------------
  
  # 6. average ionic radius
  v.append(sum([el.average_ionic_radius*fr for (el,fr) in zip(elem,fracs)]))

  # 7. average max oxidation state
  v.append(sum([el.max_oxidation_state*fr for (el,fr) in zip(elem,fracs)]))

  # 8. average max oxidation state
  v.append(sum([el.min_oxidation_state*fr for (el,fr) in zip(elem,fracs)]))

  return v