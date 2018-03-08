# @Date:   11-10-2017
# @Last modified time: 11-10-2017
# @License: GNU Public License v3

#from: https://stackoverflow.com/questions/5141437/filtering-os-walk-dirs-and-files
import fnmatch
import os
import os.path
import re

includes = ['*.jpg', '*.png'] # for files only
#excludes = ['faces'] # for dirs and files

# transform glob patterns to regular expressions
includes = r'|'.join([fnmatch.translate(x) for x in includes])
#excludes = r'|'.join([fnmatch.translate(x) for x in excludes]) or r'$.'

for root, dirs, files in os.walk('.'):

    # exclude dirs
    #dirs[:] = [os.path.join(root, d) for d in dirs]
    #dirs[:] = [d for d in dirs if not re.match(excludes, d)]

    # exclude/include files
    files = [os.path.join(root, f) for f in files]
    #files = [f for f in files if not re.match(excludes, f)]
    files = [f for f in files if re.match(includes, f)]

    for fname in files:
        print(fname)
