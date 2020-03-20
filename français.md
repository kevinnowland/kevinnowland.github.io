---
layout: default
title: Kevin Nowland French Blog
---

# Français

Bonjour, le Monde !

<ul>
  {% for post in site.categories.french %}
    <li>
      <a href="{{ post.url }}">{{ post.date | date_to_string }} - {{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
