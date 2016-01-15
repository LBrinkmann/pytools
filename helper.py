def print_kernel_connection():
    import json
    import os
    import urllib
    import IPython
    import re
    from IPython.lib import kernel
    connection_file_path = kernel.get_connection_file()
    connection_file = os.path.basename(connection_file_path)
    kernel_id = connection_file.split('-', 1)[1].split('.')[0]


    r = urllib.request.urlopen('http://127.0.0.1:8888/api/sessions')
    sessions = json.loads(r.read().decode(r.info().get_param('charset') or 'utf-8'))
    for sess in sessions:
        if sess['kernel']['id'] == kernel_id:
            jpath = sess['notebook']['path']
            break

    name = re.search('.*/(.*).ipynb', jpath).group(1)

    print('PROMPT_COMMAND=\'echo -ne "\\033]0; %s \\007"\''%name)
    print('ssh levin@10.0.50.40 -t \'ipython console --existing %s \''%(connection_file_path))
    print('')
