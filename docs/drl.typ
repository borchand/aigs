#import "@local/lilka:0.0.0": *
#import "@preview/lovelace:0.3.0": *
#import "@preview/wrap-it:0.1.1": wrap-content
#show: lilka

#set text(size: 12pt)
#set par(justify: true, leading: 1.2em)

#align(center, text(1.5em, tracking: 0.1em, upper[Deep Reinforcement Learning]))
#v(2em)

Quality diversity algorithms, rather than finding a specific good solution to a
problem, construct archives of different high performing solutions. "Different"
can be defined in various ways, an obvious example being "solutions that look
different to a vision model" @kumarAutomatingSearchArtificial2024. A
popular#footnote[At least in this lab] quality diversity algorithm is the
pedagogically named Multi-dimensional Archive of Phenotypic Elites (MAP-Elites)
@mouret2015. Think of it as having a fitness dimension, on which we maximize,
and (a) behavior dimension(s) which we want to populate.

= Content generation

We can optimize levels, just like we can optimize players. You must now:
- Define a fitness function for a level
- Define a behavior space for a level



#[
  #show heading.where(level: 1): set heading(numbering: none)
  = Index of Sources <touying:unoutlined>
  #bibliography("zotero.bib", title: none, style: "ieee")
]
