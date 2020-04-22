#!/bin/bash

set -e

echo "========= Pulling Github Page ========="

mkdir ../page

git clone git@github.com:Oceanbao/oceanbao.github.io.git ../page

rm -rf ../page/*/
rm -rf ../page/*

echo "========= Copying Public to Page ========="

hugo -t sam

cp -r public/* ../page

rm -rf public

echo "========= Updating Page Repo ========="

cd ../page 

git add -A && git commit -m "`date`: updated blog" && git push

cd ..

rm -rf page

echo "===========Fin de programme=========="
