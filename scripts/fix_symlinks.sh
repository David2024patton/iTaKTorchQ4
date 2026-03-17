#!/bin/bash
cd ~/torch_lib
for f in lib*.so; do
    link="${f%.so}.so.0"
    if [ ! -e "$link" ]; then
        ln -sf "$f" "$link"
        echo "Created symlink: $link -> $f"
    fi
done
echo "All symlinks:"
ls -la *.so.0
