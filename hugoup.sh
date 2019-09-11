#!/bin/bash

echo "========= Updating Blog Repo ========="

cd /home/oceanbao/Git-Page/blog

git add -A && git commit && git push

echo "========= Pulling Github Page ========="
git clone git@github.com:Oceanbao/oceanbao.github.io.git ../page

rm -rf ../page/*/
rm -rf ../page/*

echo "========= Copying Public to Page ========="

hugo -t sam

cp -rf public/* ../page

rm -rf public 

echo "========= Updating Page Repo ========="
cd ../page 

git add -A && git commit && git push

cd ..

rm -rf page

echo "===========Fin de programme=========="
