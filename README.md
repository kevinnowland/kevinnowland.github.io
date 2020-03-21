# A Github Website

Going to make an attempt at hosting a website on github.

## Changing a notebook to a post


The zeroth step is to have a notebook with an appropriate layout
and categories (code). Then conver to markdown with

```bash
cd _notebooks/
jupyter nbconvert --to markdown --no-prompt notebook.ipynb 
mv notebook.md ../_posts/
```

If the notebook has images move them to the images folder.
```bash
mv image_folder ../assets/images/
```
Then you have to rename where the images are coming from. So
inside the markdwn file, run a command like
```
% s/notebook_files/\/assets\/images\/notebook_files/g
```

Finally, if you did `df.head()` anywhere, clear the border
so the formatting works by running
```
% s/ border=\"1\"//g
```




## Selecting code themes

To change code themes

```bash
rougify style <theme-name> >> assets/css/syntax.css
```

To get a list of theme names,
```bash
rougify help style
```
