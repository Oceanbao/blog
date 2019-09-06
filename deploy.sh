#!/bin/bash

echo -e "\033[0;32mDeploying updates to GitHub...\033[0m"


rm public/* -rf

# Build the project.
hugo -t sam # if using a theme, replace with `hugo -t <YOURTHEME>`



# Go To Public folder
cd public
# Add changes to git.
git add .

# Commit changes.
msg="rebuilding site `date`"
if [ $# -eq 1 ]
  then msg="$1"
fi

git commit -m "$msg"

# Push source and build repos.
git push origin master

# Come Back up to the Project Root
cd ..


git add .
git commit -m "Update blog `date`"
git push origin master
