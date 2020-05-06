#!/bin/bash

set -e

printf "\033[0;32mDeploying updates to GitHub...\033[0m\n"


echo "========= Building hugo site =========" 
hugo -t sam


echo "========= Updating Github page repo (public/) =========" 
# update `public` upstream git repo
cd public

git add .

# Commit changes.
msg="rebuilding site $(date)"
if [ -n "$*" ]; then
	msg="$*"
fi
git commit -m "$msg"

# Push source and build repos.
git push origin master

echo "========= Updating blog repo ========="

cd ..

git add .

# Commit changes.
msg="rebuilding site $(date)"
if [ -n "$*" ]; then
	msg="$*"
fi
git commit -m "$msg"

# Push source and build repos.
git push origin master

echo "========= FIN de CODE=========" 
ls -l public
