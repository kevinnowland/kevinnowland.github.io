---
layout: default
title: Kevin Nowland's homepage
published: 20 March 2020
updated: 23 March 2020
---


# Home


This website is a repository of the things that I'm currently thinking about
and working on. Most of the content on this website is organized into a set of
blogs, each of which pertains to some interest that I am currently exploring
or have worked through in the past.


## Transparency


This website exists as a work in progress. The differences between what 
exists on my laptop and what you see here should be minimal. This means
that some posts will be drafts, although I'll tag them as such.

As I am writing this on 23 March 2020, this work in progress design is by
necessity given that I want to have this website available and I only began
writing it last week.



## Further behind the curtain


If you really want to know how this works technically or why I'm makig some
of the choices that I am, read this section.


### Code stuff


More technically, this website exists as a 
[git repository](https://github.com/kevinnowland/kevinnowland.github.io).
Feel free to take a look at anything there. Since GitHub Pages uses Jekyll,
most of the content exists as markdown files that I edit using vim. With
the exception of the syntax highlighting (generated using 
`rougify style <theme>`), I am writing the CSS and html layouts.

Posts on the <a class="inline" href="/code">code</a> blog are formed by 
converting Jupyter notebooks to markdown using 
`jupyter nbconvert --to markdown`. I then make some light substitutions to 
the resultant markdown file and move move included figures to the appropriate 
folder. I need to write a fish script to makes these formulaic changes for me,
but have not yet. This means that code posts might not be visible as drafts 
the way other posts should be.

The work in progress nature of the webste happens because I am by and
large committing directly to master and immediately pushing up to GitHub--
as long as jekyll doesn't tell me something is completely broken. This
isn't a collaborative project, so I'm not going to feel bad about not putting 
changes on separate branches.


### Design choices


As with all personal websites, this site is a public mask over a private life.
What's here is what I want to show publicly, but I hope it doesn't come off
too much as a technical portfolio, even if that is partly its purpose.
I mostly hope to use this site to encourage my own growth. If I am not 
making progress in some interest, then that will be reflected by 
a lack of change here, and I don't want my public persona to solidify. 
This is another reason to minimize lag between what exists locally on
my laptop and what is pushed to GitHub.
