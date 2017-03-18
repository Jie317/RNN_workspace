#! /bin/bash

# Script to samplify the commit and push steps. All the arguments are concatenated as the commmit message.

git add -A
if [ $# -eq 0 ]; then
	echo "Warning: Better to leave a message to this commit. Using default message: update"
	echo ""
	git commit -m "Update"
else
	git commit -m "$*"
fi
git push origin master
