#!/bin/bash
BASEDIR=$(dirname "$0")
# setup DIV2KCustom
DIR="$BASEDIR/DIV2K"
if [ -d "$DIR" ]; then
  echo "Downloading dataset to ${DIR}"
else
  echo "Directory $DIR doesn't exist, I will create it"
  mkdir $DIR
  echo "Downloading dataset to ${DIR}"
fi


wget http://oukei.huakunshen.com:8000/DIV2KCustom/same_300.zip -O same_300.zip
rm -rf $DIR/same_300
unzip 'same_300.zip' -d $DIR
rm 'same_300.zip'

wget http://oukei.huakunshen.com:8000/DIV2KCustom/same.zip -O same.zip
rm -rf $DIR/same
unzip 'same.zip' -d $DIR
rm 'same.zip'

wget http://oukei.huakunshen.com:8000/DIV2KCustom/diff.zip -O diff.zip
rm -rf $DIR/diff
unzip 'diff.zip' -d $DIR
rm 'diff.zip'

# setup TEXT
DIR="$BASEDIR/TEXT"
if [ -d "$DIR" ]; then
  echo "Downloading dataset to ${DIR}"
else
  echo "Directory $DIR doesn't exist, I will create it"
  mkdir $DIR
  echo "Downloading dataset to ${DIR}"
fi
wget http://oukei.huakunshen.com:8000/TEXT/diff.zip -O diff.zip
rm -rf $DIR/diff
unzip 'diff.zip' -d $DIR
rm 'diff.zip'

wget http://oukei.huakunshen.com:8000/TEXT/same.zip -O same.zip
rm -rf $DIR/same
unzip 'same.zip' -d $DIR
rm 'same.zip'

