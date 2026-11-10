#!/bin/bash
# usage: usurvey.sh DIR... ; prints element:U pairs from every INCAR (POSCAR/CONTCAR line 6 gives the order)
for root in "$@"; do
  for x in $(find $root -maxdepth 3 -name INCAR -not -path "*stage1*" 2>/dev/null); do
    d=$(dirname $x); p=$d/POSCAR; [ -f $p ] || p=$d/CONTCAR; [ -f $p ] || continue
    els=$(sed -n 6p $p); us=$(grep -h "^ *LDAUU" $x | head -1 | sed 's/.*=//'); ls_=$(grep -h "^ *LDAUL" $x | head -1 | sed 's/.*=//')
    [ -z "$us" ] && { echo "NOU"; continue; }
    set -- $els; e=("$@"); set -- $us; u=("$@"); set -- $ls_; l=("$@")
    for i in "${!e[@]}"; do echo "${e[$i]}:${u[$i]}:${l[$i]}"; done
  done
done | sort | uniq -c
