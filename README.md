# A Github Website

Going to make an attempt at hosting a website on github.

## Changing a notebook to a post


The zeroth step is to have a notebook with an appropriate layout
Then convert to markdown with

```bash
cd code/_notebooks/
jupyter nbconvert --to markdown --no-prompt notebook.ipynb 
mv notebook.md ../_posts/YYYY-MM-DD-notebook.md
```

If the notebook has images move them to the images folder.
```bash
mv notebook_files/ ../../assets/images/
```
Then you have to rename where the images are coming from. So
inside the markdwn file, run a command like
```
% s/notebook_files//\/assets\/images\/notebook_files/g
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

I've since edited `syntax.css` to make it look more like a jupyter notebook
output.



## Ruby stuff



Ruby frustrates me so much compared to python. This is being
updated for instructions used on Ubuntu 20.04.1.
I got `rbenv` installed just using `apt` and then installed
`ruby-build` into the `~/.rbenv/plugins` directory just by cloning it there.

I then did 
```bash
rbenv install 2.7.0
rbenv global 2.7.0
```

For consistency sake with earlier README versions I
ran 
```bash
gem install bundler -v 2.1.4
```
but the command
```
bundler install
```
in this repo failed because of an ffi verison.
So then I ran
```bash
sudo apt-get install libffi-dev
```
and then I could `bundler install`.
