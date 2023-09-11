import pickle
import os
from datetime import datetime, date, time

from external.pybullet_planning.pybullet_tools.utils import join_paths, get_parent_dir

def data_saver(composite_path, opt):
    # path = join_paths(get_parent_dir(__file__), os.pardir, '.')
    # dbfile = open(path+'/experiments/run_'+str(opt.env_type)+'_{}'.format(datetime.now()), 'ab')
    dbfile = open('./experiments/run_' + str(opt.env_type) + '_{}'.format(datetime.now()), 'ab')
    db = {}
    db['composite_path'] = composite_path
    db['opt'] = opt
    pickle.dump(db, dbfile)
    dbfile.close()