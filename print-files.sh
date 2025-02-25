#!/bin/bash

for file in "$@"
do
    echo
    echo "# $file"
    echo
    cat "$file"
    echo
done

# #!/bin/bash

# # Directory to search for .py files (current directory by default)
# DIR=${1:-.}

# # Find all Python files in the specified directory
# for file in "$DIR"/*.py; do
#   if [[ -f "$file" ]]; then
#     # Print the filename as a header
#     echo -e "### $(basename "$file")\n"
    
#     # Print the contents of the file
#     cat "$file"
    
#     # Add a blank line for separation
#     echo -e "\n"
#   fi
# done