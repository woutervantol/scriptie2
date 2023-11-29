


def p_to_path(p):
    return f"{p['flamingo_path']}/{p['simsize']}/{p['cosmology']}"


def p_to_filename(p):
    return f"{p['simsize']}_{p['cosmology']}_{p['selection_type_name']}_res{p['resolution']}"



def get_nth_newest_file(path, n):
    import os
    search_dir = path
    os.chdir(search_dir)
    files = filter(os.path.isfile, os.listdir(search_dir))
    files = [os.path.join(search_dir, f) for f in files] # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x))
    return files[-n]

