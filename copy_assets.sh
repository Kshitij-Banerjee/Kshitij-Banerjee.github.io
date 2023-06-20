git status  | grep md | cut -d ' ' -f2 | xargs cat | grep png | awk -F'[()]' '{print $2}' | awk -F'/' '{print $NF}' | while read -r filename; do cp "/Users/kay/Library/Mobile Documents/iCloud~com~logseq~logseq/Documents/assets/$filename" "./static"; done  
echo "copy images done"
