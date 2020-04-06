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



## Ruby stuff


I am not a ruby programmer, I am leaving some notes to 
myself that document how I got jekyll setup on my mac
when I had been doing things on ubuntu.

Assuming you have `rbenv` installed, make sure ruby 2.7.0.
I had to update the available versions with (on mac, at least)
```bash
brew upgrade ruby-build
```
On my ubuntu machine I'm not sure what would be required. I then ran
```bash
rbenv install 2.7.0
rbenv global 2.7.0
```
to install and make this version the global version. The next
step was to install the proper version of bundler, which is 2.1.4 (as
of this writing, this is the stable version).
```bash
gem install bundler 
```
Then the jekyll server got started with
```bash
bundle exec jekyll serve
```
In this repository, I ran
```bash
bundler install
```
because it knows how to use the Gemfile. I think.
