for name in ./*.bmp; do
    convert -resize 32*32\! $name $name
done

