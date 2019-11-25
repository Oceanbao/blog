#!/bin/bash

echo "========= Updating Blog Repo ========="

cd /home/oceanbao/HACKING/GPage/blog

~/bin/gitup.sh

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

~/bin/gitup.sh

cd ..

rm -rf page

echo "===========Fin de programme=========="
