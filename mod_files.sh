# change any excalamation links to, pdflinks
git s | grep md | awk -F' ' '{ print $2 }' | xargs -I{} sed -i '' 's|!\[\(.*\)\](\(.*\.pdf\))|{{< pdflink "\2" "\1" >}}|g' "{}"

# Modify the image files to remove asset prefixes, as images are globaly available instatic
# git s | grep md | awk -F' ' '{ print $2 }' | xargs -I{} sed -i '' 's|\.\./assets/|/|g' "{}" 

# Update image paths and convert Markdown image tags to GLightbox shortcode
git s | grep '\.md' | awk -F' ' '{ print $2 }' | xargs -I{} sed -i '' \
-e 's|!\[\(.*\)\](\.\./assets/\([^)]*\))|{{< glightbox href="/\2" src="/\2" alt="\1" >}}|g' \
-e 's|!\[\(.*\)\](\(/[^)]*\))|{{< glightbox href="\2" src="\2" alt="\1" >}}|g' \
"{}"
