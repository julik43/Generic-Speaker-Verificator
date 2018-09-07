#!/bin/bash

usage (){
	echo "Usage: bash change_corpus_sr.sh corpus_location new_corpus_location sample_rate"
}


if [ -z "$1" ]
then
        echo "No corpus location was given."
        usage
        exit
fi

if [ -z "$2" ]
then
        echo "No new corpus location was given."
        usage
        exit
fi



DIRS=$1/*
for d in $DIRS
do
  if [ -d "$d" ]
  then
    echo "Processing 1st level $d ..."
    
    DIRS2=$d/*
    for d2 in $DIRS2
    do
      if [ -d "$d2" ]
      then
        echo "Processing 2nd level $d2 ..."

        DIRS3=$d2/*
        for d3 in $DIRS3
        do
          if [ -d "$d3" ]
          then
            echo "Processing 3rd level $d3 ..."
            NEWDIRBASE="$2/$(basename $d)/$(basename $d2)/$(basename $d3)"
            mkdir -p $NEWDIRBASE

            FILES=$d3/*.m4a
            #echo $FILES
            for f in $FILES
            do
              echo "Processing file $f ..."
              new_f=${f/%m4a/flac}
              avconv -i "$f" "$new_f"
              
              NEWF="$NEWDIRBASE/$(basename $new_f)"
              echo "   moving to $NEWF ..."
              mv "$new_f" $NEWF
            done
          fi
        done
      fi
    done
  fi
done

echo "done."
